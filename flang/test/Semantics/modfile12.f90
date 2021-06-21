! RUN: %S/test_modfile.sh %s %t %flang_fc1
! REQUIRES: shell
module m
  integer(8), parameter :: a = 1, b = 2_8
  parameter(n=3,l=-3,e=1.0/3.0)
  real :: x(a:2*(a+b*n)-1)
  real, dimension(8) :: y
  type t(c, d)
    integer, kind :: c = 1
    integer, len :: d = a + b
  end type
  type(t(a+3,:)), allocatable :: z
  class(t(a+4,:)), allocatable :: z2
  class(*), allocatable :: z4
  real*2 :: f
  complex*32 :: g
  type t2(i, j, h)
    integer, len :: h
    integer, kind :: j
    integer, len :: i
  end type
contains
  subroutine foo(x)
    real :: x(2:)
  end
  subroutine bar(x)
    real :: x(..)
  end
  subroutine baz(x)
    type(*) :: x
  end
end

!Expect: m.mod
!module m
!  integer(8),parameter::a=1_8
!  integer(8),parameter::b=2_8
!  integer(4),parameter::n=3_4
!  integer(4),parameter::l=-3_4
!  real(4),parameter::e=3.333333432674407958984375e-1_4
!  real(4)::x(1_8:13_8)
!  real(4)::y(1_8:8_8)
!  type::t(c,d)
!    integer(4),kind::c=1_4
!    integer(4),len::d=3_4
!  end type
!  type(t(c=4_4,d=:)),allocatable::z
!  class(t(c=5_4,d=:)),allocatable::z2
!  class(*),allocatable::z4
!  real(2)::f
!  complex(16)::g
!  type::t2(i,j,h)
!    integer(4),len::h
!    integer(4),kind::j
!    integer(4),len::i
!  end type
!contains
!  subroutine foo(x)
!    real(4)::x(2_8:)
!  end
!  subroutine bar(x)
!    real(4)::x(..)
!  end
!  subroutine baz(x)
!    type(*)::x
!  end
!end
