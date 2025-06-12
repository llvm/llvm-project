! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

! Test that checks whether compiler directives can be inlined without mistaking it as comment.

module m
contains
#define MACRO(X)  subroutine func1(X); real(2) :: X; !dir$ ignore_tkr(d) X; end subroutine func1;
MACRO(foo)

!CHECK: SUBROUTINE func1 (foo)
!CHECK: !DIR$ IGNORE_TKR (d) foo
!CHECK: END SUBROUTINE func1

  subroutine func2(foo)
    real(2) :: foo; !dir$ ignore_tkr(d) foo;
  end subroutine func2

!CHECK: SUBROUTINE func2 (foo)
!CHECK: !DIR$ IGNORE_TKR (d) foo
!CHECK: END SUBROUTINE func2

  subroutine func3(foo)
    real(2) :: foo; !dir$ ignore_tkr(d) foo; end subroutine func3;

!CHECK: SUBROUTINE func3 (foo)
!CHECK: !DIR$ IGNORE_TKR (d) foo
!CHECK: END SUBROUTINE func3

end module
