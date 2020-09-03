! RUN: %S/test_modfile.sh %s %t %f18
module m
  character(2), parameter :: prefix = 'c_'
  integer, bind(c, name='c_a') :: a
  procedure(sub), bind(c, name=prefix//'b'), pointer :: b
  type, bind(c) :: t
    real :: c
  end type
  real :: d
  external :: d
  bind(c, name='dd') :: d
  real :: e
  bind(c, name='ee') :: e
  external :: e
  bind(c, name='ff') :: f
  real :: f
  external :: f
contains
  subroutine sub() bind(c, name='sub')
  end
end

!Expect: m.mod
!module m
!  character(2_4,1),parameter::prefix="c_"
!  integer(4),bind(c, name="c_a")::a
!  procedure(sub),bind(c, name="c_b"),pointer::b
!  type,bind(c)::t
!    real(4)::c
!  end type
!  procedure(real(4)),bind(c, name="dd")::d
!  procedure(real(4)),bind(c, name="ee")::e
!  procedure(real(4)),bind(c, name="ff")::f
!contains
!  subroutine sub() bind(c, name="sub")
!  end
!end
