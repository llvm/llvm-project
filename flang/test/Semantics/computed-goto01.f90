! RUN: %python %S/test_errors.py %s %flang_fc1
! Check that a basic computed goto compiles

INTEGER, DIMENSION (2) :: B

GOTO (100) 1
GOTO (100) I
GOTO (100) I+J
GOTO (100) B(1)

GOTO (100, 200) 1
GOTO (100, 200) I
GOTO (100, 200) I+J
GOTO (100, 200) B(1)

GOTO (100, 200, 300) 1
GOTO (100, 200, 300) I
GOTO (100, 200, 300) I+J
GOTO (100, 200, 300) B(1)

100 CONTINUE
200 CONTINUE
300 CONTINUE
END
