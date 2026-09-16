! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPpointer_forall_degenerated_assignment() {
! CHECK: %[[P:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
! CHECK: hlfir.declare %[[P]] {{.*}}pointer{{.*}} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield %{{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield %{{.*}} : i32
! CHECK: }  (%{{.*}}: i32) {
! CHECK: }

subroutine pointer_forall_degenerated_assignment()
  integer, pointer :: p
  integer, target :: t(1)
  forall (i=1:1)
     ! Test hits a TODO when uncommented.
     ! p => t(i)
  end forall
end subroutine

! CHECK-LABEL: func.func @_QPlogical_forall_degenerated_assignment() {
! CHECK:         %[[L_ALLOCA:.*]] = fir.alloca !fir.logical<4> {bindc_name = "l", uniq_name = "_QFlogical_forall_degenerated_assignmentEl"}
! CHECK:         %[[L:.*]]:2 = hlfir.declare %[[L_ALLOCA]] {uniq_name = "_QFlogical_forall_degenerated_assignmentEl"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:         hlfir.forall lb {
! CHECK:           hlfir.yield %{{.*}} : i32
! CHECK:         } ub {
! CHECK:           hlfir.yield %{{.*}} : i32
! CHECK:         }  (%{{.*}}: i32) {
! CHECK:           hlfir.region_assign {
! CHECK:             %[[TRUE:.*]] = arith.constant true
! CHECK:             %[[VAL:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
! CHECK:             hlfir.yield %[[VAL]] : !fir.logical<4>
! CHECK:           } to {
! CHECK:             hlfir.yield %[[L]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:           }
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine logical_forall_degenerated_assignment()
  logical :: l
  forall (i=1:1)
    l = .true.
  end forall
end subroutine
