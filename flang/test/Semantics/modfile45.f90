! RUN: %python %S/test_modfile.py %s %flang_fc1
! Ensures that uninitialized allocatable components in a structure constructor
! appear with explicit NULL() in the expression representation.
module m
  type t
    real, allocatable :: x1, x2, x3
  end type
  type t2
    type(t) :: a = t(NULL(),x2=NULL())
  end type
end module

!Expect: m.mod
!module m
!type::t
!real(4),allocatable::x1
!real(4),allocatable::x2
!real(4),allocatable::x3
!end type
!type::t2
!type(t)::a=t(x1=NULL(),x2=NULL(),x3=NULL())
!end type
!intrinsic::null
!end
