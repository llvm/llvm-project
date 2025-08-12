! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Tests for the putenv intrinsics.

subroutine bad_kind_error(str, status)
  CHARACTER(len=255) :: str
  INTEGER(2) :: status
  !ERROR: Actual argument for 'status=' has bad type or kind 'INTEGER(2)'
  call putenv(str, status)
end subroutine bad_kind_error

subroutine bad_args_error()
  !ERROR: missing mandatory 'str=' argument
  call putenv()
end subroutine bad_args_error

subroutine bad_function(str)
  CHARACTER(len=255) :: str
  INTEGER :: status
  call putenv(str, status)
  !ERROR: Cannot call subroutine 'putenv' like a function
  status = putenv(str)
end subroutine bad_function

subroutine bad_sub(str)
  CHARACTER(len=255) :: str
  INTEGER :: status
  status = putenv(str)
  !ERROR: Cannot call function 'putenv' like a subroutine
  call putenv(str, status)
end subroutine bad_sub

subroutine good_subroutine(str, status)
  CHARACTER(len=255) :: str
  INTEGER :: status
  call putenv(str, status)
end subroutine good_subroutine

subroutine good_function(str, status)
  CHARACTER(len=255) :: str
  INTEGER :: status
  status = putenv(str)
end subroutine good_function
