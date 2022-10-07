! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s

! Tests various aspect of the lowering of polymorphic entities.

module polymorphic_test
  type p1
    integer :: a
    integer :: b
  end type

  type, extends(p1) :: p2
    real :: c
  end type

  contains

  ! Test correct access to polymorphic entity component.
  subroutine component_access(p)
    class(p1) :: p
    print*, p%a
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPcomponent_access(
! CHECK-SAME: %[[P:.*]]: !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> {fir.bindc_name = "p"}) {
! CHECK: %[[FIELD:.*]] = fir.field_index a, !fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>
! CHECK: %[[COORD:.*]] = fir.coordinate_of %[[P]], %[[FIELD]] : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK: %[[LOAD:.*]] = fir.load %[[COORD]] : !fir.ref<i32>
! CHECK: %{{.*}} = fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[LOAD]]) : (!fir.ref<i8>, i32) -> i1

end module
