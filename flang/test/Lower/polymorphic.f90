! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s

! Tests various aspect of the lowering of polymorphic entities.

module polymorphic_test
  type p1
    integer :: a
    integer :: b
  contains
    procedure :: print
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

  subroutine print(this)
    class(p1) :: this
  end subroutine

  ! Test embox of fir.type to fir.class to be passed-object.
  subroutine check()
    type(p1) :: t1
    type(p2) :: t2
    call t1%print()
    call t2%print()
  end subroutine

! CHECK-LABEL: func.func @_QMpolymorphic_testPcheck()
! CHECK: %[[DT1:.*]] = fir.alloca !fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}> {bindc_name = "t1", uniq_name = "_QMpolymorphic_testFcheckEt1"}
! CHECK: %[[DT2:.*]] = fir.alloca !fir.type<_QMpolymorphic_testTp2{a:i32,b:i32,c:f32}> {bindc_name = "t2", uniq_name = "_QMpolymorphic_testFcheckEt2"}
! CHECK: %[[CLASS1:.*]] = fir.embox %[[DT1]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>
! CHECK: fir.call @_QMpolymorphic_testPprint(%[[CLASS1]]) : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> ()
! CHECK: %[[BOX2:.*]] = fir.embox %[[DT2]] : (!fir.ref<!fir.type<_QMpolymorphic_testTp2{a:i32,b:i32,c:f32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp2{a:i32,b:i32,c:f32}>>
! CHECK: %[[CLASS2:.*]] = fir.convert %[[BOX2]] : (!fir.class<!fir.type<_QMpolymorphic_testTp2{a:i32,b:i32,c:f32}>>) -> !fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>> 
! CHECK: fir.call @_QMpolymorphic_testPprint(%[[CLASS2]]) : (!fir.class<!fir.type<_QMpolymorphic_testTp1{a:i32,b:i32}>>) -> ()
end module
