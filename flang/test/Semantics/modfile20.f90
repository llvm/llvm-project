! RUN: %S/test_modfile.sh %s %t %flang_fc1
! REQUIRES: shell
! Test modfiles for entities with initialization
module m
  integer, parameter :: k8 = 8
  integer(8), parameter :: k4 = k8/2
  integer, parameter :: k1 = 1
  integer(k8), parameter :: i = 2_k8
  real :: r = 2.0_k4
  character(10, kind=k1) :: c = k1_"asdf"
  character(10), parameter :: c2 = k1_"qwer"
  complex*16, parameter :: z = (1.0_k8, 2.0_k8)
  complex*16, parameter :: zn = (-1.0_k8, 2.0_k8)
  type t
    integer :: a = 123
    type(t), pointer :: b => null()
  end type
  type(t), parameter :: x = t(456)
  type(t), parameter :: y = t(789, null())
end

!Expect: m.mod
!module m
!  integer(4),parameter::k8=8_4
!  integer(8),parameter::k4=4_8
!  integer(4),parameter::k1=1_4
!  integer(8),parameter::i=2_8
!  real(4)::r
!  character(10_4,1)::c
!  character(10_4,1),parameter::c2="qwer      "
!  complex(8),parameter::z=(1._8,2._8)
!  complex(8),parameter::zn=(-1._8,2._8)
!  type::t
!    integer(4)::a=123_4
!    type(t),pointer::b=>NULL()
!  end type
!  intrinsic::null
!  type(t),parameter::x=t(a=456_4,b=NULL())
!  type(t),parameter::y=t(a=789_4,b=NULL())
!end
