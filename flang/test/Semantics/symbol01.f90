! RUN: %S/test_symbols.sh %s %t %flang_fc1
! REQUIRES: shell
! Test that intent-stmt and subprogram prefix and suffix are resolved.

!DEF: /m Module
module m
 !DEF: /m/f PRIVATE, PURE, RECURSIVE (Function) Subprogram REAL(4)
 private :: f
contains
 !DEF: /m/s BIND(C), PUBLIC, PURE (Subroutine) Subprogram
 !DEF: /m/s/x INTENT(IN) (Implicit) ObjectEntity REAL(4)
 !DEF: /m/s/y INTENT(INOUT) (Implicit) ObjectEntity REAL(4)
 pure subroutine s (x, y) bind(c)
  !REF: /m/s/x
  intent(in) :: x
  !REF: /m/s/y
  intent(inout) :: y
 contains
  !DEF: /m/s/ss PURE (Subroutine) Subprogram
  pure subroutine ss
  end subroutine
 end subroutine
 !REF: /m/f
 !DEF: /m/f/x ALLOCATABLE ObjectEntity REAL(4)
 recursive pure function f() result(x)
  !REF: /m/f/x
  real, allocatable :: x
  !REF: /m/f/x
  x = 1.0
 end function
end module
