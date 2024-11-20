! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Tests for the GETCWD intrinsics

subroutine bad_kind_error(cwd, status)
  CHARACTER(len=255) :: cwd
  INTEGER(2) :: status
  !ERROR: Actual argument for 'status=' has bad type or kind 'INTEGER(2)'
  call getcwd(cwd, status)
end subroutine bad_kind_error
  
subroutine bad_args_error()
  !ERROR: missing mandatory 'c=' argument
  call getcwd()
end subroutine bad_args_error

subroutine bad_apply_form(cwd)
  CHARACTER(len=255) :: cwd
  INTEGER :: status
  !Declaration of 'getcwd'
  call getcwd(cwd, status)
  !ERROR: Cannot call subroutine 'getcwd' like a function
  status = getcwd(cwd)
end subroutine bad_apply_form

subroutine good_subroutine(cwd, status)
  CHARACTER(len=255) :: cwd
  INTEGER :: status
  call getcwd(cwd, status)
end subroutine good_subroutine

subroutine good_function(cwd, status)
  CHARACTER(len=255) :: cwd
  INTEGER :: status
  status = getcwd(cwd)
end subroutine good_function