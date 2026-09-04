! RUN: %python %S/test_errors.py %s %flang_fc1
! Test semantic checking of F_C_STRING from ISO_C_BINDING

program test
  use iso_c_binding
  implicit none

  character(len=20) :: str
  character(len=:), allocatable :: result
  logical :: flag
  integer :: n
  real :: x
  character(len=20), dimension(2) :: str_array

  ! Valid usages
  result = f_c_string('hello')
  result = f_c_string(str)
  result = f_c_string(str, .true.)
  result = f_c_string(str, .false.)
  result = f_c_string(str, flag)
  result = f_c_string(string=str)
  result = f_c_string(string=str, asis=.true.)
  result = f_c_string(asis=.false., string=str)

  ! Invalid: missing required argument
  !ERROR: missing mandatory 'string=' argument
  result = f_c_string()

  ! Invalid: non-character first argument
  !ERROR: Actual argument for 'string=' has bad type 'INTEGER(4)'
  result = f_c_string(n)

  ! Invalid: non-character first argument (real)
  !ERROR: Actual argument for 'string=' has bad type 'REAL(4)'
  result = f_c_string(x)

  ! Invalid: non-logical second argument
  !ERROR: Actual argument for 'asis=' has bad type 'INTEGER(4)'
  result = f_c_string(str, n)

  ! Invalid: too many arguments
  !ERROR: too many actual arguments for intrinsic '__builtin_f_c_string'
  result = f_c_string(str, .true., .false.)

  ! Invalid: array argument (must be scalar)
  !ERROR: 'string=' argument has unacceptable rank 1
  result = f_c_string(str_array)

end program
