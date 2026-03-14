! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
! Check that we get portability warning for the extension:
!  - exponent-letter 'Q'

subroutine s
  real :: realvar1 = 4.0
  real :: realvar2 = 4.0D6
  real :: realvar3 = 4.0_8
  real :: realvar4 = 4.0E6_4
  real :: realvar5 = 4.0E6_8
  !PORTABILITY: nonstandard usage: Q exponent
  real :: realvar6 = 4.0Q6
end subroutine s
