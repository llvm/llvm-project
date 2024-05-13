! RUN: %python %S/test_errors.py %s %flang_fc1
MODULE test
SAVE
CONTAINS
PURE FUNCTION pf( )
   IMPLICIT NONE
   INTEGER :: pf
   INTEGER :: mc
   !OK: SAVE statement is not inherited by the function
END FUNCTION

PURE FUNCTION pf2( )
   IMPLICIT NONE
   SAVE
   INTEGER :: pf2
   !ERROR: A pure subprogram may not have a variable with the SAVE attribute
   INTEGER :: mc
END FUNCTION

! This same subroutine appears in test save02.f90 where it is not an
! error due to -fno-automatic.
SUBROUTINE foo
  INTEGER, TARGET :: t
  !ERROR: An initial data target may not be a reference to an object 't' that lacks the SAVE attribute
  INTEGER, POINTER :: p => t
end

END MODULE

