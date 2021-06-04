! RUN: %S/test_errors.sh %s %t %flang_fc1
! Check that expressions are analyzed in data statements

subroutine s1
  type :: t
    character(1) :: c
  end type
  type(t) :: x
  !ERROR: Value in structure constructor of type INTEGER(4) is incompatible with component 'c' of type CHARACTER(KIND=1,LEN=1_8)
  data x /t(1)/
end

subroutine s2
  real :: x1, x2
  integer :: i1, i2
  !ERROR: Unsupported REAL(KIND=99)
  data x1 /1.0_99/
  !ERROR: Unsupported REAL(KIND=99)
  data x2 /-1.0_99/
  !ERROR: INTEGER(KIND=99) is not a supported type
  data i1 /1_99/
  !ERROR: INTEGER(KIND=99) is not a supported type
  data i2 /-1_99/
end

subroutine s3
  complex :: z1, z2
  !ERROR: Unsupported REAL(KIND=99)
  data z1 /(1.0, 2.0_99)/
  !ERROR: Unsupported REAL(KIND=99)
  data z2 /-(1.0, 2.0_99)/
end
