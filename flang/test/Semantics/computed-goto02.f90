! RUN: %python %S/test_errors.py %s %flang_fc1
! Check that computed goto express must be a scalar integer expression
! TODO: PGI, for example, accepts a float & converts the value to int.

REAL R
COMPLEX Z
LOGICAL L
INTEGER, DIMENSION (2) :: B

!ERROR: Must have INTEGER type, but is REAL(4)
GOTO (100) 1.5
!ERROR: Must have INTEGER type, but is LOGICAL(4)
GOTO (100) .TRUE.
!ERROR: Must have INTEGER type, but is REAL(4)
GOTO (100) R
!ERROR: Must have INTEGER type, but is COMPLEX(4)
GOTO (100) Z
!ERROR: Must be a scalar value, but is a rank-1 array
GOTO (100) B

100 CONTINUE

END
