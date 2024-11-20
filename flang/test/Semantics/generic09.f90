! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
module m1
  type foo
    integer n
    integer :: m = 1
  end type
end

module m2
  use m1
  interface foo
    module procedure f1
  end interface
 contains
  type(foo) function f1(a)
    real, intent(in) :: a
    f1%n = a
    f1%m = 2
  end
end

module m3
  use m2
  interface foo
    module procedure f2
  end interface
 contains
  type(foo) function f2(a)
    double precision, intent(in) :: a
    f2%n = a
    f2%m = 3
  end
end

program main
  use m3
  type(foo) x
!CHECK: foo(n=1_4,m=1_4)
  x = foo(1)
  print *, x
!CHECK: f1(2._4)
  x = foo(2.)
  print *, x
!CHECK: f2(3._8)
  x = foo(3.d0)
  print *, x
end
