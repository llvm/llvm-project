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

! Directive between two procedures in a plain interface
module m3
  interface
    subroutine s1()
    end subroutine
!CHECK: warning: Unrecognized compiler directive was ignored
    !dir$ unknown_directive
    subroutine s2()
    end subroutine
  end interface
end module

! Generic (named) interface
module m4
  interface iface
    subroutine sub1(x)
      real, intent(in) :: x
    end subroutine sub1
!CHECK: warning: Unrecognized compiler directive was ignored
    !dir$ unknown_directive
    subroutine sub2(x)
      integer, intent(in) :: x
    end subroutine sub2
  end interface iface
end module

! Abstract interface
module m5
  abstract interface
!CHECK: warning: Unrecognized compiler directive was ignored
    !dir$ unknown_directive
    subroutine abs_sub(x)
      real, intent(in) :: x
    end subroutine abs_sub
  end interface
end module
