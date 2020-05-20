! RUN: %S/test_modfile.sh %s %t %f18
! Resolution of generic names in expressions.
! Test by using generic function in a specification expression that needs
! to be written to a .mod file.

! Resolve based on number of arguments
module m1
  interface f
    pure integer(8) function f1(x)
      real, intent(in) :: x
    end
    pure integer(8) function f2(x, y)
      real, intent(in) :: x, y
    end
    pure integer(8) function f3(x, y, z, w)
      real, intent(in) :: x, y, z, w
      optional :: w
    end
  end interface
contains
  subroutine s1(x, z)
    real :: z(f(x))  ! resolves to f1
  end
  subroutine s2(x, y, z)
    real :: z(f(x, y))  ! resolves to f2
  end
  subroutine s3(x, y, z, w)
    real :: w(f(x, y, z))  ! resolves to f3
  end
  subroutine s4(x, y, z, w, u)
    real :: u(f(x, y, z, w))  ! resolves to f3
  end
end
!Expect: m1.mod
!module m1
! interface f
!  procedure :: f1
!  procedure :: f2
!  procedure :: f3
! end interface
! interface
!  pure function f1(x)
!   real(4), intent(in) :: x
!   integer(8) :: f1
!  end
! end interface
! interface
!  pure function f2(x, y)
!   real(4), intent(in) :: x
!   real(4), intent(in) :: y
!   integer(8) :: f2
!  end
! end interface
! interface
!  pure function f3(x, y, z, w)
!   real(4), intent(in) :: x
!   real(4), intent(in) :: y
!   real(4), intent(in) :: z
!   real(4), intent(in), optional :: w
!   integer(8) :: f3
!  end
! end interface
!contains
! subroutine s1(x, z)
!  real(4) :: x
!  real(4) :: z(1_8:f1(x))
! end
! subroutine s2(x, y, z)
!  real(4) :: x
!  real(4) :: y
!  real(4) :: z(1_8:f2(x, y))
! end
! subroutine s3(x, y, z, w)
!  real(4) :: x
!  real(4) :: y
!  real(4) :: z
!  real(4) :: w(1_8:f3(x, y, z))
! end
! subroutine s4(x, y, z, w, u)
!  real(4) :: x
!  real(4) :: y
!  real(4) :: z
!  real(4) :: w
!  real(4) :: u(1_8:f3(x, y, z, w))
! end
!end

! Resolve based on type or kind
module m2
  interface f
    pure integer(8) function f_real4(x)
      real(4), intent(in) :: x
    end
    pure integer(8) function f_real8(x)
      real(8), intent(in) :: x
    end
    pure integer(8) function f_integer(x)
      integer, intent(in) :: x
    end
  end interface
contains
  subroutine s1(x, y)
    real(4) :: x
    real :: y(f(x))  ! resolves to f_real4
  end
  subroutine s2(x, y)
    real(8) :: x
    real :: y(f(x))  ! resolves to f_real8
  end
  subroutine s3(x, y)
    integer :: x
    real :: y(f(x))  ! resolves to f_integer
  end
end
!Expect: m2.mod
!module m2
! interface f
!  procedure :: f_real4
!  procedure :: f_real8
!  procedure :: f_integer
! end interface
! interface
!  pure function f_real4(x)
!   real(4), intent(in) :: x
!   integer(8) :: f_real4
!  end
! end interface
! interface
!  pure function f_real8(x)
!   real(8), intent(in) :: x
!   integer(8) :: f_real8
!  end
! end interface
! interface
!  pure function f_integer(x)
!   integer(4), intent(in) :: x
!   integer(8) :: f_integer
!  end
! end interface
!contains
! subroutine s1(x, y)
!  real(4) :: x
!  real(4) :: y(1_8:f_real4(x))
! end
! subroutine s2(x, y)
!  real(8) :: x
!  real(4) :: y(1_8:f_real8(x))
! end
! subroutine s3(x, y)
!  integer(4) :: x
!  real(4) :: y(1_8:f_integer(x))
! end
!end

! Resolve based on rank
module m3a
  interface f
    procedure :: f_elem
    procedure :: f_vector
  end interface
contains
  pure integer(8) elemental function f_elem(x) result(result)
    real, intent(in) :: x
    result = 1_8
  end
  pure integer(8) function f_vector(x) result(result)
    real, intent(in) :: x(:)
    result = 2_8
  end
end
!Expect: m3a.mod
!module m3a
! interface f
!  procedure :: f_elem
!  procedure :: f_vector
! end interface
!contains
! elemental pure function f_elem(x) result(result)
!  real(4), intent(in) :: x
!  integer(8) :: result
! end
! pure function f_vector(x) result(result)
!  real(4), intent(in) :: x(:)
!  integer(8) :: result
! end
!end

module m3b
use m3a
contains
  subroutine s1(x, y)
    real :: x
    real :: y(f(x))  ! resolves to f_elem
  end
  subroutine s2(x, y)
    real :: x(10)
    real :: y(f(x))  ! resolves to f_vector (preferred over elemental one)
  end
  subroutine s3(x, y)
    real :: x(10, 10)
    real :: y(ubound(f(x), 1))  ! resolves to f_elem
  end
end
!Expect: m3b.mod
!module m3b
! use m3a, only: f
! use m3a, only: f_elem
! use m3a, only: f_vector
!contains
! subroutine s1(x, y)
!  real(4) :: x
!  real(4) :: y(1_8:f_elem(x))
! end
! subroutine s2(x, y)
!  real(4) :: x(1_8:10_8)
!  real(4) :: y(1_8:f_vector(x))
! end
! subroutine s3(x, y)
!  real(4) :: x(1_8:10_8, 1_8:10_8)
!  real(4) :: y(1_8:10_8)
! end
!end

! Resolve defined unary operator based on type
module m4
  interface operator(.foo.)
    pure integer(8) function f_real(x)
      real, intent(in) :: x
    end
    pure integer(8) function f_integer(x)
      integer, intent(in) :: x
    end
  end interface
contains
  subroutine s1(x, y)
    real :: x
    real :: y(.foo. x)  ! resolves to f_real
  end
  subroutine s2(x, y)
    integer :: x
    real :: y(.foo. x)  ! resolves to f_integer
  end
end
!Expect: m4.mod
!module m4
! interface operator(.foo.)
!  procedure :: f_real
!  procedure :: f_integer
! end interface
! interface
!  pure function f_real(x)
!   real(4), intent(in) :: x
!   integer(8) :: f_real
!  end
! end interface
! interface
!  pure function f_integer(x)
!   integer(4), intent(in) :: x
!   integer(8) :: f_integer
!  end
! end interface
!contains
! subroutine s1(x, y)
!  real(4) :: x
!  real(4) :: y(1_8:f_real(x))
! end
! subroutine s2(x, y)
!  integer(4) :: x
!  real(4) :: y(1_8:f_integer(x))
! end
!end

! Resolve defined binary operator based on type
module m5
  interface operator(.foo.)
    pure integer(8) function f1(x, y)
      real, intent(in) :: x
      real, intent(in) :: y
    end
    pure integer(8) function f2(x, y)
      real, intent(in) :: x
      complex, intent(in) :: y
    end
  end interface
contains
  subroutine s1(x, y)
    complex :: x
    real :: y(1.0 .foo. x)  ! resolves to f2
  end
  subroutine s2(x, y)
    real :: x
    real :: y(1.0 .foo. x)  ! resolves to f1
  end
end
!Expect: m5.mod
!module m5
! interface operator(.foo.)
!  procedure :: f1
!  procedure :: f2
! end interface
! interface
!  pure function f1(x, y)
!   real(4), intent(in) :: x
!   real(4), intent(in) :: y
!   integer(8) :: f1
!  end
! end interface
! interface
!  pure function f2(x, y)
!   real(4), intent(in) :: x
!   complex(4), intent(in) :: y
!   integer(8) :: f2
!  end
! end interface
!contains
! subroutine s1(x, y)
!  complex(4) :: x
!  real(4) :: y(1_8:f2(1._4, x))
! end
! subroutine s2(x, y)
!  real(4) :: x
!  real(4) :: y(1_8:f1(1._4, x))
! end
!end
