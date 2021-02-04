! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

! Check the analyzed form of a defined operator or assignment.

! Type-bound defined assignment
module m1
  type :: t
  contains
    procedure :: b1 => s1
    procedure, pass(y) :: b2 => s2
    generic :: assignment(=) => b1, b2
  end type
contains
  subroutine s1(x, y)
    class(t), intent(out) :: x
    integer, intent(in) :: y
  end
  subroutine s2(x, y)
    real, intent(out) :: x
    class(t), intent(in) :: y
  end
  subroutine test1(x)
    type(t) :: x
    real :: a
    !CHECK: CALL s1(x,1_4)
    x = 1
    !CHECK: CALL s2(a,x)
    a = x
  end
  subroutine test2(x)
    class(t) :: x
    real :: a
    !CHECK: CALL x%b1(1_4)
    x = 1
    !CHECK: CALL x%b2(a)
    a = x
  end
end

! Type-bound operator
module m2
  type :: t2
  contains
    procedure, pass(x2) :: b2 => f
    generic :: operator(+) => b2
  end type
contains
  integer pure function f(x1, x2)
    class(t2), intent(in) :: x1
    class(t2), intent(in) :: x2
  end
  subroutine test2(x, y)
    class(t2) :: x
    type(t2) :: y
    !CHECK: i=f(x,y)
    i = x + y
    !CHECK: i=x%b2(y)
    i = y + x
  end
end module

! Non-type-bound assignment and operator
module m3
  type t
  end type
  interface assignment(=)
    subroutine s1(x, y)
      import
      class(t), intent(out) :: x
      integer, intent(in) :: y
    end
  end interface
  interface operator(+)
    integer function f(x, y)
      import
      class(t), intent(in) :: x, y
    end
  end interface
contains
  subroutine test(x, y)
    class(t) :: x, y
    !CHECK: CALL s1(x,2_4)
    x = 2
    !CHECK: i=f(x,y)
    i = x + y
  end
end

