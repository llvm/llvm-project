! RUN: %python %S/test_errors.py %s %flang_fc1
! A procedure (EXTERNAL/INTRINSIC) may not have a DATA-style initializer.
! The initializer used to be silently accepted and dropped (#222168).
subroutine s1
  external foo
  !ERROR: Procedure 'foo' may not have a DATA-style initializer
  integer foo /1/
end subroutine
subroutine s2
  intrinsic sin
  !ERROR: Procedure 'sin' may not have a DATA-style initializer
  integer sin /1/
end subroutine
