! RUN: %S/test_errors.sh %s %t %f18
! Check that non-logical expressions are not allowed.
! Check that non-scalar expressions are not allowed.
! TODO: Insure all non-logicals are prohibited.

LOGICAL, DIMENSION (2) :: B

!ERROR: Must have LOGICAL type, but is REAL(4)
IF (A) A = LOG (A)
!ERROR: Must be a scalar value, but is a rank-1 array
IF (B) A = LOG (A)

END
