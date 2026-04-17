! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Namelist processing deferred under -pedantic. Therefore,
! x should be implicitly (default) typed as a real, and i
! should be implicitly typed as an integer (because it starts
! with i-n). Both should print a type error at their explicit
! declarations which must agree with their implicit typing.
subroutine s 
  !PORTABILITY: Namelist processing not deferred [-Wnamelist-no-defer]
  namelist /nl/ x, i 
  !ERROR: The type of 'x' has already been implicitly declared as REAL(4)
  integer :: x 
  !ERROR: The type of 'i' has already been implicitly declared as INTEGER(4)
  real :: i
end subroutine 
