!RUN: %python %S/test_errors.py %s %flang_fc1

! Test that POINTER with PARAMETER doesn't crash.

subroutine s1
  !ERROR: 'a' may not have both the POINTER and PARAMETER attributes
  pointer a
  !ERROR: PARAMETER attribute not allowed on 'a'
  parameter(a=3)
end subroutine

subroutine s2
  !ERROR: 'b' may not have both the POINTER and PARAMETER attributes
  integer, pointer :: b
  !ERROR: PARAMETER attribute not allowed on 'b'
  parameter(b=3)
end subroutine

