! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Tests for the HOSTNM intrinsics.

subroutine bad_kind_error(cwd, status)
  CHARACTER(len=255) :: cwd
  INTEGER(2) :: status
  !ERROR: Actual argument for 'status=' has bad type or kind 'INTEGER(2)'
  call hostnm(cwd, status)
end subroutine bad_kind_error

subroutine bad_args_error()
  !ERROR: missing mandatory 'c=' argument
  call hostnm()
end subroutine bad_args_error

subroutine bad_function(cwd)
  CHARACTER(len=255) :: cwd
  INTEGER :: status
  call hostnm(cwd, status)
  !ERROR: Cannot call subroutine 'hostnm' like a function
  status = hostnm(cwd)
end subroutine bad_function

subroutine bad_sub(cwd)
  CHARACTER(len=255) :: cwd
  INTEGER :: status
  status = hostnm(cwd)
  !ERROR: Cannot call function 'hostnm' like a subroutine
  call hostnm(cwd, status)
end subroutine bad_sub

subroutine good_subroutine(cwd, status)
  CHARACTER(len=255) :: cwd
  INTEGER :: status
  call hostnm(cwd, status)
end subroutine good_subroutine

subroutine good_function(cwd, status)
  CHARACTER(len=255) :: cwd
  INTEGER :: status
  status = hostnm(cwd)
end subroutine good_function
