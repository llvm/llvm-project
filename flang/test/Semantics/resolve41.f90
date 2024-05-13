! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  implicit none
  real, parameter :: a = 8.0
  !ERROR: Must have INTEGER type, but is REAL(4)
  integer :: aa = 2_a
  integer :: b = 8
  ! C713 A scalar-int-constant-name shall be a named constant of type integer.
  !ERROR: Must be a constant value
  integer :: bb = 2_b
  !TODO: should get error -- not scalar
  !integer, parameter :: c(10) = 8
  !integer :: cc = 2_c
  integer, parameter :: d = 47
  !ERROR: INTEGER(KIND=47) is not a supported type
  integer :: dd = 2_d
  !ERROR: Parameter 'e' not found
  integer :: ee = 2_e
  !ERROR: Missing initialization for parameter 'f'
  integer, parameter :: f
  integer :: ff = 2_f
  !ERROR: REAL(KIND=23) is not a supported type
  real(d/2) :: g
  !ERROR: REAL*47 is not a supported type
  real*47 :: h
  !ERROR: COMPLEX*47 is not a supported type
  complex*47 :: i
end
