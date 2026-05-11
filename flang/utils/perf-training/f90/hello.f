! RUN: %flang -c %s
! RUN: %flang_skip_driver -c %s

      PROGRAM HELLO
        IMPLICIT NONE
        INTEGER I
        INTEGER NUM
        CHARACTER ARG * 32

        NUM = 0
        CALL GETARG(1, ARG)
        IF (LEN_TRIM(ARG) .GT. 0) THEN
          READ (ARG, *, IOSTAT = I) NUM
        END IF
        IF (NUM .GT. 0) THEN
          DO 10 I = 1, NUM
            WRITE (*, 100) I
10        CONTINUE
        ELSE
          WRITE (*, 200)
        END IF
100     FORMAT(' ', I3, '. Hello')
200     FORMAT(' Hello, world!')
      END PROGRAM HELLO
