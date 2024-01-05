! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! This test checks the lowering of atomic read

program acc_atomic_test
  real g, h
  !$acc atomic read
     g = h
end program acc_atomic_test

! CHECK: func @_QQmain() attributes {fir.bindc_name = "acc_atomic_test"} {
! CHECK: %[[VAR_G:.*]] = fir.alloca f32 {bindc_name = "g", uniq_name = "_QFEg"}
! CHECK: %[[G_DECL:.*]]:2 = hlfir.declare %[[VAR_G]] {uniq_name = "_QFEg"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK: %[[VAR_H:.*]] = fir.alloca f32 {bindc_name = "h", uniq_name = "_QFEh"}
! CHECK: %[[H_DECL:.*]]:2 = hlfir.declare %[[VAR_H]] {uniq_name = "_QFEh"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK: acc.atomic.read %[[G_DECL]]#1 = %[[H_DECL]]#1 : !fir.ref<f32>, f32
! CHECK: return
! CHECK: }

! Test lowering atomic read for pointer variables.
! Please notice to use %[[VAL_4]] and %[[VAL_1]] for operands of atomic
! operation, instead of %[[VAL_3]] and %[[VAL_0]].

subroutine atomic_read_pointer()
  integer, pointer :: x, y

  !$acc atomic read
    y = x

  x = y
end

! CHECK-LABEL: func.func @_QPatomic_read_pointer() {
! CHECK:   %[[X:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "x", uniq_name = "_QFatomic_read_pointerEx"}
! CHECK:   %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFatomic_read_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:   %[[Y:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "y", uniq_name = "_QFatomic_read_pointerEy"}
! CHECK:   %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFatomic_read_pointerEy"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:   %[[LOAD_X:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:   %[[BOX_ADDR_X:.*]] = fir.box_addr %[[LOAD_X]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:   %[[LOAD_Y:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:   %[[BOX_ADDR_Y:.*]] = fir.box_addr %[[LOAD_Y]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:   acc.atomic.read %[[BOX_ADDR_Y]] = %[[BOX_ADDR_X]] : !fir.ptr<i32>, i32
! CHECK: }

subroutine atomic_read_with_convert()
  integer(4) :: x
  integer(8) :: y

  !$acc atomic read
  y = x
end

! CHECK-LABEL: func.func @_QPatomic_read_with_convert() {
! CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFatomic_read_with_convertEx"}
! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFatomic_read_with_convertEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[Y:.*]] = fir.alloca i64 {bindc_name = "y", uniq_name = "_QFatomic_read_with_convertEy"}
! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFatomic_read_with_convertEy"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK: %[[CONV:.*]] = fir.convert %[[X_DECL]]#1 : (!fir.ref<i32>) -> !fir.ref<i64>
! CHECK: acc.atomic.read %[[Y_DECL]]#1 = %[[CONV]] : !fir.ref<i64>, i32
