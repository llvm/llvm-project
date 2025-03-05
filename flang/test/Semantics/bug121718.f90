! RUN: %flang_fc1 2>&1 | FileCheck %s --allow-empty
! CHECK-NOT: error
! Regression test simplified from LLVM bug 121718.
! Ensure no crash and no spurious error message.
module m1
  type foo
    integer x
  end type
 contains
  subroutine test
    print *, foo(123)
  end
end
module m2
  interface foo
    procedure f
  end interface
  type foo
    real x
  end type
 contains
  complex function f(x)
    complex, intent(in) :: x
    f = x
  end
end
program main
  use m1
  use m2
  call test
end
