! RUN: bbc -fopenacc -emit-fir -hlfir=false %s -o - | FileCheck %s

! This test checks the lowering of atomic read

!CHECK: func @_QQmain() attributes {fir.bindc_name = "acc_atomic_test"} {
!CHECK: %[[VAR_G:.*]] = fir.alloca f32 {bindc_name = "g", uniq_name = "_QFEg"}
!CHECK: %[[VAR_H:.*]] = fir.alloca f32 {bindc_name = "h", uniq_name = "_QFEh"}
!CHECK: acc.atomic.read %[[VAR_G]] = %[[VAR_H]] : !fir.ref<f32>, f32
!CHECK: return
!CHECK: }

program acc_atomic_test
    real g, h
    !$acc atomic read
       g = h
end program acc_atomic_test

! Test lowering atomic read for pointer variables.
! Please notice to use %[[VAL_4]] and %[[VAL_1]] for operands of atomic
! operation, instead of %[[VAL_3]] and %[[VAL_0]].

!CHECK-LABEL: func.func @_QPatomic_read_pointer() {
! CHECK:   %[[X:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "x", uniq_name = "_QFatomic_read_pointerEx"}
! CHECK:   %[[Y:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "y", uniq_name = "_QFatomic_read_pointerEy"}
! CHECK:   %[[LOAD_X:.*]] = fir.load %[[X]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:   %[[BOX_ADDR_X:.*]] = fir.box_addr %[[LOAD_X]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:   %[[LOAD_Y:.*]] = fir.load %[[Y]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:   %[[BOX_ADDR_Y:.*]] = fir.box_addr %[[LOAD_Y]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:   acc.atomic.read %[[BOX_ADDR_Y]] = %[[BOX_ADDR_X]] : !fir.ptr<i32>, i32
! CHECK: }

subroutine atomic_read_pointer()
  integer, pointer :: x, y

  !$acc atomic read
    y = x

  x = y
end

subroutine atomic_read_with_convert()
  integer(4) :: x
  integer(8) :: y

  !$acc atomic read
  y = x
end

! CHECK-LABEL: func.func @_QPatomic_read_with_convert() {
! CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFatomic_read_with_convertEx"}
! CHECK: %[[Y:.*]] = fir.alloca i64 {bindc_name = "y", uniq_name = "_QFatomic_read_with_convertEy"}
! CHECK: %[[CONV:.*]] = fir.convert %[[X]] : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK: acc.atomic.read %[[Y]] = %[[CONV]] : !fir.ref<i64>, i32
