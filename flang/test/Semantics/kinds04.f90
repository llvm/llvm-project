! RUN: %python %S/test_errors.py %s %flang_fc1
! C716 If both kind-param and exponent-letter appear, exponent-letter 
! shall be E.
! C717 The value of kind-param shall specify an approximation method that 
! exists on the processor.
subroutine s(var)
  real :: realvar1 = 4.0E6_4
  real :: realvar2 = 4.0D6
  real :: realvar3 = 4.0Q6
  real :: realvar4 = 4.0D6_8
  real :: realvar5 = 4.0Q6_16
  real :: realvar6 = 4.0E6_8
  real :: realvar7 = 4.0E6_10
  real :: realvar8 = 4.0E6_16
  !ERROR: Unsupported REAL(KIND=32)
  real :: realvar9 = 4.0E6_32

  double precision :: doublevar1 = 4.0E6_4
  double precision :: doublevar2 = 4.0D6
  double precision :: doublevar3 = 4.0Q6
  double precision :: doublevar4 = 4.0D6_8
  double precision :: doublevar5 = 4.0Q6_16
  double precision :: doublevar6 = 4.0E6_8
  double precision :: doublevar7 = 4.0E6_10
  double precision :: doublevar8 = 4.0E6_16
  !ERROR: Unsupported REAL(KIND=32)
  double precision :: doublevar9 = 4.0E6_32
end subroutine s
