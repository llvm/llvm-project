! Test EXIT with dynamically optional arguments.
! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPexit_opt_dummy(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "status", fir.optional}) {
subroutine exit_opt_dummy(status)
  integer, optional :: status
  call exit(status)
! CHECK:  %[[VAL_1:.*]] = fir.is_present %[[VAL_0]] : (!fir.ref<i32>) -> i1
! CHECK:  %[[VAL_2:.*]] = fir.if %[[VAL_1]] -> (i32) {
! CHECK:    %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:    fir.result %[[VAL_3]] : i32
! CHECK:  } else {
! CHECK:    %[[VAL_4:.*]] = arith.constant 0 : i32
! CHECK:    fir.result %[[VAL_4]] : i32
! CHECK:  }
! CHECK:  %[[VAL_5:.*]] = fir.call @_FortranAExit(%[[VAL_6:.*]]) {{.*}}: (i32) -> none
end subroutine

! CHECK-LABEL: func @_QPexit_pointer(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "status"}) {
subroutine exit_pointer(status)
  integer, pointer :: status
  call exit(status)
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ptr<i32>) -> i64
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_5:.*]] = arith.cmpi ne, %[[VAL_3]], %[[VAL_4]] : i64
! CHECK:  %[[VAL_6:.*]] = fir.if %[[VAL_5]] -> (i32) {
! CHECK:    %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:    %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:    %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ptr<i32>
! CHECK:    fir.result %[[VAL_9]] : i32
! CHECK:  } else {
! CHECK:    %[[VAL_10:.*]] = arith.constant 0 : i32
! CHECK:    fir.result %[[VAL_10]] : i32
! CHECK:  }
! CHECK:  %[[VAL_11:.*]] = fir.call @_FortranAExit(%[[VAL_12:.*]]) {{.*}}: (i32) -> none
end subroutine
