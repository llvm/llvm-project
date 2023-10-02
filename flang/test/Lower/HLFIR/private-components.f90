! Test that private component names are mangled inside fir.record
! in a way that allow components with the same name to be added in
! type extensions.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module name_clash
  type:: t
    integer, private :: i
  end type
  type(t), parameter :: cst = t(42)
end module

!CHECK-LABEL: func.func @_QPuser_clash(
!CHECK-SAME: !fir.ref<!fir.type<_QFuser_clashTt2{_QMname_clashTt.i:i32,i:i32}>>
!CHECK-SAME: !fir.ref<!fir.type<_QMname_clashTt{_QMname_clashTt.i:i32}>>
subroutine user_clash(a, at)
  use name_clash
  type,extends(t) :: t2
    integer :: i = 2
  end type
  type(t2) :: a, b
  type(t) :: at
  print *, a%i
  print *, t2(t=at)
  a = b
end subroutine

! CHECK-LABEL: func.func @_QPclash_with_intrinsic_module(
! CHECK-SAME: !fir.ref<!fir.type<_QFclash_with_intrinsic_moduleTmy_class{_QMieee_arithmeticTieee_class_type.which:i8,which:i8}>>
subroutine clash_with_intrinsic_module(a)
 use ieee_arithmetic
 type, extends(ieee_class_type) :: my_class
    integer(1) :: which
 end type
 type(my_class) :: a
end subroutine
