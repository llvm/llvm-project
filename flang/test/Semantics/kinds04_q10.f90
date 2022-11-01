! RUN: %python %S/test_errors.py %s %flang_fc1 -triple x86_64-unknown-linux-gnu
! C716 If both kind-param and exponent-letter appear, exponent-letter
! shall be E. (As an extension we also allow an exponent-letter which matches
! the kind-param)
! C717 The value of kind-param shall specify an approximation method that
! exists on the processor.
!
! This test is for x86_64, where exponent-letter 'q' is for
! 10-byte extended precision
! UNSUPPORTED: system-windows

subroutine s(var)
  real :: realvar1 = 4.0E6_4
  real :: realvar2 = 4.0D6
  real :: realvar3 = 4.0Q6
  real :: realvar4 = 4.0D6_8
  real :: realvar5 = 4.0Q6_10
  !WARNING: Explicit kind parameter on real constant disagrees with exponent letter 'q'
  real :: realvar6 = 4.0Q6_16
  real :: realvar7 = 4.0E6_8
  real :: realvar8 = 4.0E6_10
  real :: realvar9 = 4.0E6_16
  !ERROR: Unsupported REAL(KIND=32)
  real :: realvar10 = 4.0E6_32

  double precision :: doublevar1 = 4.0E6_4
  double precision :: doublevar2 = 4.0D6
  double precision :: doublevar3 = 4.0Q6
  double precision :: doublevar4 = 4.0D6_8
  !WARNING: Explicit kind parameter on real constant disagrees with exponent letter 'q'
  double precision :: doublevar5 = 4.0Q6_16
  double precision :: doublevar6 = 4.0E6_8
  double precision :: doublevar7 = 4.0E6_10
  double precision :: doublevar8 = 4.0E6_16
  !ERROR: Unsupported REAL(KIND=32)
  double precision :: doublevar9 = 4.0E6_32
end subroutine s
