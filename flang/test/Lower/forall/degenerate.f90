! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPpointer_forall_degenerated_assignment() {

subroutine pointer_forall_degenerated_assignment()
  integer, pointer :: p
  integer, target :: t(1)
  forall (i=1:1)
     ! Test hits a TODO when uncommented.
     ! p => t(i)
  end forall
end subroutine

! CHECK-LABEL: func @_QPlogical_forall_degenerated_assignment() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.logical<4> {bindc_name = "l", uniq_name = "_QFlogical_forall_degenerated_assignmentEl"}
! CHECK:         %[[VAL_2:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:         fir.do_loop %[[VAL_7:.*]] = %[[VAL_3]] to %[[VAL_5]] step %[[VAL_6]] unordered {
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (index) -> i32
! CHECK:           fir.store %[[VAL_8]] to %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_9:.*]] = arith.constant true
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i1) -> !fir.logical<4>
! CHECK:           fir.store %[[VAL_10]] to %[[VAL_1]] : !fir.ref<!fir.logical<4>>
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine logical_forall_degenerated_assignment()
  logical :: l
  forall (i=1:1)
    l = .true.
  end forall
end subroutine

