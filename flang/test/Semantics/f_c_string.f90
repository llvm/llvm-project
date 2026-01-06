! RUN: %python %S/test_errors.py %s %flang_fc1
! Test semantic checking of F_C_STRING from ISO_C_BINDING

program test
  use iso_c_binding
  implicit none
  
  character(len=20) :: str
  character(len=:), allocatable :: result
  logical :: flag
  integer :: n
  
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
  !ERROR: Dummy argument 'string=' (#1) is not OPTIONAL and is not associated with an actual argument in this procedure reference
  result = f_c_string()
  
  ! Invalid: too many arguments
  !ERROR: Too many actual arguments (3) passed to procedure that expects only 2
  result = f_c_string(str, .true., .false.)
  
  ! Invalid: non-character first argument
  !ERROR: Actual argument type 'INTEGER(4)' is not compatible with dummy argument type 'CHARACTER(KIND=1,LEN=*)'
  result = f_c_string(n)
  
  ! Invalid: non-logical second argument
  !ERROR: Actual argument type 'INTEGER(4)' is not compatible with dummy argument type 'LOGICAL(4)'
  result = f_c_string(str, n)
  
end program
