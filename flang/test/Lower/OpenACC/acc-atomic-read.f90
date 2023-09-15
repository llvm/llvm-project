! RUN: bbc --use-desc-for-alloc=false -fopenacc -emit-fir %s -o - | FileCheck %s

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
!CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "x", uniq_name = "_QFatomic_read_pointerEx"}
!CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.ptr<i32> {uniq_name = "_QFatomic_read_pointerEx.addr"}
!CHECK:         %[[VAL_2:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK:         fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "y", uniq_name = "_QFatomic_read_pointerEy"}
!CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.ptr<i32> {uniq_name = "_QFatomic_read_pointerEy.addr"}
!CHECK:         %[[VAL_5:.*]] = fir.zero_bits !fir.ptr<i32>
!CHECK:         fir.store %[[VAL_5]] to %[[VAL_4]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         acc.atomic.read %[[VAL_7]] = %[[VAL_6]]   : !fir.ptr<i32>, i32
!CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ptr<i32>
!CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.ptr<i32>>
!CHECK:         fir.store %[[VAL_9]] to %[[VAL_10]] : !fir.ptr<i32>
!CHECK:         return
!CHECK:       }

subroutine atomic_read_pointer()
  integer, pointer :: x, y

  !$acc atomic read
    y = x

  x = y
end
