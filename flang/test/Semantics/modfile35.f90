! RUN: %python %S/test_modfile.py %s %flang_fc1
module m1
  type :: t1
  contains
    procedure, pass(x) :: p1 => f
    procedure, non_overridable :: p2 => f
    procedure, nopass :: p3 => f
    generic :: operator(+) => p1
    generic :: operator(-) => p2
    generic :: operator(<) => p1
    generic :: operator(.and.) => p2
  end type
contains
  integer(8) pure function f(x, y)
    class(t1), intent(in) :: x
    integer, intent(in) :: y
  end
  ! Operators resolve to type-bound operators in t1
  subroutine test1(x, y, a, b)
    class(t1) :: x
    integer :: y
    real :: a(x + y)
    real :: b(x .lt. y)
  end
  ! Operators resolve to type-bound operators in t1, compile-time resolvable
  subroutine test2(x, y, a, b)
    class(t1) :: x
    integer :: y
    real :: a(x - y)
    real :: b(x .and. y)
  end
  ! Operators resolve to type-bound operators in t1, compile-time resolvable
  subroutine test3(x, y, a)
    type(t1) :: x
    integer :: y
    real :: a(x + y)
  end
end
!Expect: m1.mod
!module m1
! type :: t1
! contains
!  procedure, pass(x) :: p1 => f
!  procedure, non_overridable :: p2 => f
!  procedure, nopass :: p3 => f
!  generic :: operator(+) => p1
!  generic :: operator(-) => p2
!  generic :: operator(<) => p1
!  generic :: operator(.and.) => p2
! end type
!contains
! pure function f(x, y)
!  class(t1), intent(in) :: x
!  integer(4), intent(in) :: y
!  integer(8) :: f
! end
! subroutine test1(x, y, a, b)
!  class(t1) :: x
!  integer(4) :: y
!  real(4) :: a(1_8:x%p1(y))
!  real(4) :: b(1_8:x%p1(y))
! end
! subroutine test2(x, y, a, b)
!  class(t1) :: x
!  integer(4) :: y
!  real(4) :: a(1_8:f(x, y))
!  real(4) :: b(1_8:f(x, y))
! end
! subroutine test3(x,y,a)
!  type(t1) :: x
!  integer(4) :: y
!  real(4) :: a(1_8:f(x,y))
! end
!end

module m2
  type :: t1
  contains
    procedure, pass(x) :: p1 => f1
    generic :: operator(+) => p1
  end type
  type, extends(t1) :: t2
  contains
    procedure, pass(y) :: p2 => f2
    generic :: operator(+) => p2
  end type
contains
  integer(8) pure function f1(x, y)
    class(t1), intent(in) :: x
    integer, intent(in) :: y
  end
  integer(8) pure function f2(x, y)
    class(t1), intent(in) :: x
    class(t2), intent(in) :: y
  end
  subroutine test1(x, y, a)
    class(t1) :: x
    integer :: y
    real :: a(x + y)
  end
  ! Resolve to operator in parent class
  subroutine test2(x, y, a)
    class(t2) :: x
    integer :: y
    real :: a(x + y)
  end
  ! 2nd arg is passed object
  subroutine test3(x, y, a)
    class(t1) :: x
    class(t2) :: y
    real :: a(x + y)
  end
end
!Expect: m2.mod
!module m2
! type :: t1
! contains
!  procedure, pass(x) :: p1 => f1
!  generic :: operator(+) => p1
! end type
! type, extends(t1) :: t2
! contains
!  procedure, pass(y) :: p2 => f2
!  generic :: operator(+) => p2
! end type
!contains
! pure function f1(x, y)
!  class(t1), intent(in) :: x
!  integer(4), intent(in) :: y
!  integer(8) :: f1
! end
! pure function f2(x, y)
!  class(t1), intent(in) :: x
!  class(t2), intent(in) :: y
!  integer(8) :: f2
! end
! subroutine test1(x, y, a)
!  class(t1) :: x
!  integer(4) :: y
!  real(4) :: a(1_8:x%p1(y))
! end
! subroutine test2(x, y, a)
!  class(t2) :: x
!  integer(4) :: y
!  real(4) :: a(1_8:x%p1(y))
! end
! subroutine test3(x, y, a)
!  class(t1) :: x
!  class(t2) :: y
!  real(4) :: a(1_8:y%p2(x))
! end
!end

module m3
  type :: t1
  contains
    procedure, pass(x) :: p1 => f1
    procedure :: p3 => f3
    generic :: operator(.binary.) => p1
    generic :: operator(.unary.) => p3
  end type
  type, extends(t1) :: t2
  contains
    procedure, pass(y) :: p2 => f2
    generic :: operator(.binary.) => p2
  end type
contains
  integer(8) pure function f1(x, y)
    class(t1), intent(in) :: x
    integer, intent(in) :: y
  end
  integer(8) pure function f2(x, y)
    class(t1), intent(in) :: x
    class(t2), intent(in) :: y
  end
  integer(8) pure function f3(x)
    class(t1), intent(in) :: x
  end
  subroutine test1(x, y, a)
    class(t1) :: x
    integer :: y
    real :: a(x .binary. y)
  end
  ! Resolve to operator in parent class
  subroutine test2(x, y, a)
    class(t2) :: x
    integer :: y
    real :: a(x .binary. y)
  end
  ! 2nd arg is passed object
  subroutine test3(x, y, a)
    class(t1) :: x
    class(t2) :: y
    real :: a(x .binary. y)
  end
  subroutine test4(x, y, a)
    class(t1) :: x
    class(t2) :: y
    real :: a(.unary. x + .unary. y)
  end
end
!Expect: m3.mod
!module m3
!  type::t1
!  contains
!    procedure,pass(x)::p1=>f1
!    procedure::p3=>f3
!    generic::operator(.binary.)=>p1
!    generic::operator(.unary.)=>p3
!  end type
!  type,extends(t1)::t2
!  contains
!    procedure,pass(y)::p2=>f2
!    generic::operator(.binary.)=>p2
!  end type
!contains
!  pure function f1(x,y)
!    class(t1),intent(in)::x
!    integer(4),intent(in)::y
!    integer(8)::f1
!  end
!  pure function f2(x,y)
!    class(t1),intent(in)::x
!    class(t2),intent(in)::y
!    integer(8)::f2
!  end
!  pure function f3(x)
!    class(t1),intent(in)::x
!    integer(8)::f3
!  end
!  subroutine test1(x,y,a)
!    class(t1)::x
!    integer(4)::y
!    real(4)::a(1_8:x%p1(y))
!  end
!  subroutine test2(x,y,a)
!    class(t2)::x
!    integer(4)::y
!    real(4)::a(1_8:x%p1(y))
!  end
!  subroutine test3(x,y,a)
!    class(t1)::x
!    class(t2)::y
!    real(4)::a(1_8:y%p2(x))
!  end
!  subroutine test4(x,y,a)
!    class(t1)::x
!    class(t2)::y
!    real(4)::a(1_8:x%p3()+y%p3())
!  end
!end
