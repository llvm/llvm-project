! RUN: %python %S/test_errors.py %s %flang_fc1
! A procedure (EXTERNAL/INTRINSIC) may not have a DATA-style initializer.
! The initializer used to be silently accepted and dropped (#222168), or,
! for an intrinsic that is not an unrestricted specific function, crashed
! the compiler (CHECK(designator.has_value()) in data-to-inits.cpp).
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
subroutine s3 ! used to crash: SUM is not an unrestricted specific intrinsic
  intrinsic sum
  !ERROR: Procedure 'sum' may not have a DATA-style initializer
  integer sum /1/
end subroutine
subroutine s4(f) ! dummy procedure
  external f
  !ERROR: Procedure 'f' may not have a DATA-style initializer
  integer f /1/
end subroutine
subroutine s5 ! no INTRINSIC/EXTERNAL: a plain local object named like an
  integer sum /1/ ! intrinsic must stay accepted (guard keys on symbol class)
  print *, sum
end subroutine
subroutine s6 ! procedure POINTER initialization must stay accepted (8.6.7)
  interface
    integer function tgt()
    end function
  end interface
  procedure(tgt), pointer :: p => null()
end subroutine
