! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Check that we get portability warnings for the extensions
!  - exponent-letter 'Q'
!  - matching but non-'E' exponent letter together with kind-param

subroutine s
  real :: realvar1 = 4.0
  real :: realvar2 = 4.0D6
  real :: realvar3 = 4.0_8
  real :: realvar4 = 4.0E6_4
  real :: realvar5 = 4.0E6_8
  !WARNING: nonstandard usage: Q exponent
  real :: realvar6 = 4.0Q6
  !WARNING: Explicit kind parameter together with non-'E' exponent letter is not standard
  real :: realvar7 = 4.0D6_8
  !WARNING: Explicit kind parameter on real constant disagrees with exponent letter 'd'
  real :: realvar8 = 4.0D6_4
end subroutine s
