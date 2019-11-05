import cv2

# Arquivo para identificar parte frontal do rosto
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
# Arquivo para identificar o sorriso
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# variavel para captura pela webcan
cap = cv2.VideoCapture(0)

# variavel para escrita de video
out = cv2.VideoWriter('facial-recognition.avi', cv2.VideoWriter_fourcc(*'XVID'), 24.0, (640, 480))


# plota o retangulo na tela
def retangulo(captura, coord_inicial, coord_final_x, coord_final_y, cor, espessura):
    return cv2.rectangle(captura, coord_inicial, (coord_final_x, coord_final_y), cor, espessura)


while True:
    # captura frame por frame
    ret, frame = cap.read()

    # converte para cinza - melhora o desempenho do reconhecimento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecta e especifica a escala da face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        # em cinza - com coordenadas
        zi_gray_face = gray[(y - 25):y + (h + 35), x:x + (w - 20)]  # zi = zona de interesse = rosto
        # com cor - com coordenadas
        zi_color_face = frame[(y - 25):y + (h + 35), x:x + (w - 20)]

        retangulo(frame, (x, y), (x + w), (y + h), (255, 0, 0), 2)

        # detecta e especifica a escala do sorriso
        smile = smile_cascade.detectMultiScale(zi_gray_face, scaleFactor=1.8, minNeighbors=35)

        for (a, b, c, d) in smile:
            zi_gray_smile = gray[a:a + c, b:b + d]  # zi = zona de interesse = sorriso

            retangulo(zi_color_face, (a, b), (a + c), (b + d), (255, 255, 0), 2)

            # escrevendo texto acoplado na zona de interesse(ri) do rosto
            cv2.putText(zi_color_face, 'SORRINDO', (a, d), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))

    out.write(frame)

    # exibe o resultado
    cv2.imshow('frame', frame)
    # termina o programa com a tecla 'x'
    if cv2.waitKey(20) & 0xFF == ord('x'):
        break
    elif cv2.waitKey(20) & 0xFF == ord('x'.upper()):
        break


# libera a captura
cap.release()
# libera a filmagem de video
out.release()
# destroi todas as janelas
cv2.destroyAllWindows()
