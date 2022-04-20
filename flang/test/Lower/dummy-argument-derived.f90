! Test lowering of derived type dummy arguments
! RUN: bbc -emit-fir %s -o - | FileCheck %s
module type_defs
  type simple_type
    integer :: i
  end type
  type with_kind(k)
    integer, kind :: k
    real(k) :: x
  end type
end module

! -----------------------------------------------------------------------------
!     Test passing of derived type arguments that do not require a
!     fir.box (runtime descriptor).
! -----------------------------------------------------------------------------

! Test simple type scalar with no attribute.
! CHECK-LABEL: func @_QPtest1(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.type<_QMtype_defsTsimple_type{i:i32}>> {fir.bindc_name = "a"}) {
subroutine test1(a)
  use type_defs
  type(simple_type) :: a
end subroutine

! Test simple type explicit array with no attribute.
! CHECK-LABEL: func @_QPtest2(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMtype_defsTsimple_type{i:i32}>>> {fir.bindc_name = "a"}) {
subroutine test2(a)
  use type_defs
  type(simple_type) :: a(100)
end subroutine

! Test simple type scalar with TARGET attribute.
! CHECK-LABEL: func @_QPtest3(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.type<_QMtype_defsTsimple_type{i:i32}>> {fir.bindc_name = "a", fir.target}) {
subroutine test3(a)
  use type_defs
  type(simple_type), target :: a
end subroutine

! Test simple type explicit array with TARGET attribute.
! CHECK-LABEL: func @_QPtest4(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMtype_defsTsimple_type{i:i32}>>> {fir.bindc_name = "a", fir.target}) {
subroutine test4(a)
  use type_defs
  type(simple_type), target :: a(100)
end subroutine

! Test kind parametrized derived type scalar with no attribute.
! CHECK-LABEL: func @_QPtest1k(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.type<_QMtype_defsTwith_kindK4{x:f32}>> {fir.bindc_name = "a"}) {
subroutine test1k(a)
  use type_defs
  type(with_kind(4)) :: a
end subroutine

! Test kind parametrized derived type explicit array with no attribute.
! CHECK-LABEL: func @_QPtest2k(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMtype_defsTwith_kindK4{x:f32}>>> {fir.bindc_name = "a"}) {
subroutine test2k(a)
  use type_defs
  type(with_kind(4)) :: a(100)
end subroutine

! Test kind parametrized derived type scalar with TARGET attribute.
! CHECK-LABEL: func @_QPtest3k(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.type<_QMtype_defsTwith_kindK4{x:f32}>> {fir.bindc_name = "a", fir.target}) {
subroutine test3k(a)
  use type_defs
  type(with_kind(4)), target :: a
end subroutine

! Test kind parametrized derived type explicit array with TARGET attribute.
! CHECK-LABEL: func @_QPtest4k(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<100x!fir.type<_QMtype_defsTwith_kindK4{x:f32}>>> {fir.bindc_name = "a", fir.target}) {
subroutine test4k(a)
  use type_defs
  type(with_kind(4)), target :: a(100)
end subroutine

! -----------------------------------------------------------------------------
!     Test passing of derived type arguments that require a fir.box (runtime descriptor).
! -----------------------------------------------------------------------------

! Test simple type assumed shape array with no attribute.
! CHECK-LABEL: func @_QPtest5(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMtype_defsTsimple_type{i:i32}>>> {fir.bindc_name = "a"}) {
subroutine test5(a)
  use type_defs
  type(simple_type) :: a(:)
end subroutine

! Test simple type assumed shape array with TARGET attribute.
! CHECK-LABEL: func @_QPtest6(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMtype_defsTsimple_type{i:i32}>>> {fir.bindc_name = "a", fir.target}) {
subroutine test6(a)
  use type_defs
  type(simple_type), target :: a(:)
end subroutine

! Test kind parametrized derived type assumed shape array with no attribute.
! CHECK-LABEL: func @_QPtest5k(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMtype_defsTwith_kindK4{x:f32}>>> {fir.bindc_name = "a"}) {
subroutine test5k(a)
  use type_defs
  type(with_kind(4)) :: a(:)
end subroutine

! Test kind parametrized derived type assumed shape array with TARGET attribute.
! CHECK-LABEL: func @_QPtest6k(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMtype_defsTwith_kindK4{x:f32}>>> {fir.bindc_name = "a", fir.target}) {
subroutine test6k(a)
  use type_defs
  type(with_kind(4)), target :: a(:)
end subroutine
