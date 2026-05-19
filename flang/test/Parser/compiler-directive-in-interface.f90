! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

! Test that unrecognized compiler directives inside INTERFACE blocks
! produce a warning rather than a parse error.

module m
  interface
    subroutine ccff
    end subroutine ccff
!CHECK: warning: Unrecognized compiler directive was ignored
    !dir$ id "test"
  end interface
end module

module m2
  interface
!CHECK: warning: Unrecognized compiler directive was ignored
    !dir$ id "test"
    subroutine foo(a)
      integer, intent(in) :: a
    end subroutine foo
  end interface
end module


