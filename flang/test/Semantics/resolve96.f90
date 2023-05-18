! RUN: %python %S/test_errors.py %s %flang_fc1

! Check distinguishability for specific procedures of defined operators and
! assignment. These are different from names because there a normal generic
! is invoked the same way as a type-bound generic.
! E.g. for a generic name like 'foo', the generic name is invoked as 'foo(x, y)'
! while the type-bound generic is invoked as 'x%foo(y)'.
! But for 'operator(.foo.)', it is 'x .foo. y' in either case.
! So to check the specifics of 'operator(.foo.)' we have to consider all
! definitions of it visible in the current scope.

! One operator(.foo.) comes from interface-stmt, the other is type-bound.
module m1
  type :: t1
  contains
    procedure, pass :: p => s1
    generic :: operator(.foo.) => p
  end type
  type :: t2
  end type
  !ERROR: Generic 'OPERATOR(.foo.)' may not have specific procedures 's2' and 't1%p' as their interfaces are not distinguishable
  interface operator(.foo.)
    procedure :: s2
  end interface
contains
  integer function s1(x1, x2)
    class(t1), intent(in) :: x1
    class(t2), intent(in) :: x2
  end
  integer function s2(x1, x2)
    class(t1), intent(in) :: x1
    class(t2), intent(in) :: x2
  end
end module

! assignment(=) as type-bound generic in each type
module m2
  type :: t1
    integer :: n
  contains
    procedure, pass(x1) :: p1 => s1
    !ERROR: Generic 'assignment(=)' may not have specific procedures 't1%p1' and 't2%p2' as their interfaces are not distinguishable
    generic :: assignment(=) => p1
  end type
  type :: t2
    integer :: n
  contains
    procedure, pass(x2) :: p2 => s2
    generic :: assignment(=) => p2
  end type
contains
  subroutine s1(x1, x2)
    class(t1), intent(out) :: x1
    class(t2), intent(in) :: x2
    x1%n = x2%n + 1
  end subroutine
  subroutine s2(x1, x2)
    class(t1), intent(out) :: x1
    class(t2), intent(in) :: x2
    x1%n = x2%n + 2
  end subroutine
end module
