! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine iargc_test
  implicit none
  integer :: n
  integer(4) :: i4
  character(32) :: value
  !OK:
  i4 = iargc()
  !ERROR: Cannot call function 'iargc' like a subroutine
  call iargc()
end subroutine iargc_test

subroutine getarg_test_1
  implicit none
  integer :: n
  character(32) :: value
  !OK:
  call getarg(n, value)
  !ERROR: Cannot call subroutine 'getarg' like a function
  n = getarg(1, value)
end subroutine getarg_test_1

subroutine getarg_test_2
  implicit none
  integer :: n
  character(32) :: value
  !ERROR: No explicit type declared for 'getarg'
  n = getarg(1, value)
end subroutine getarg_test_2

subroutine getarg_test_3
  implicit none
  integer :: n
  real :: r
  character(32) :: value
  integer :: bad_value
  !OK:
  call getarg(n, value)
  !ERROR: Actual argument for 'pos=' has bad type 'REAL(4)'
  call getarg(r, value)
  !ERROR: Actual argument for 'value=' has bad type 'INTEGER(4)'
  call getarg(n, bad_value)
end subroutine getarg_test_3

subroutine getarg_test_4
  implicit none
  integer(2) :: n2
  integer(8) :: n8
  character(32) :: value
  !OK:
  call getarg(n2, value)
  !OK:
  call getarg(n8, value)
end subroutine getarg_test_4
