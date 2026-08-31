!RUN: %python %S/test_errors.py %s %flang_fc1

! Test that POINTER with PARAMETER doesn't crash.

subroutine s1
  pointer a
  !ERROR: PARAMETER attribute not allowed on 'a'
  !ERROR: 'a' may not have both the POINTER and PARAMETER attributes
  parameter(a=3)
end subroutine

subroutine s2
  integer, pointer :: b
  !ERROR: 'b' may not have both the POINTER and PARAMETER attributes
  !ERROR: PARAMETER attribute not allowed on 'b'
  parameter(b=3)
end subroutine

