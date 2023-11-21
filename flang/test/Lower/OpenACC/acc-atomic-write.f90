! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! This test checks the lowering of atomic write

!CHECK: func @_QQmain() attributes {fir.bindc_name = "acc_atomic_write_test"} {
!CHECK: %[[VAR_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[VAR_X]] {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[VAR_Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[VAR_Y]] {uniq_name = "_QFEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[CONST_7:.*]] = arith.constant 7 : i32
!CHECK: {{.*}} = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
!CHECK: %[[VAR_7y:.*]] = arith.muli %[[CONST_7]], {{.*}} : i32
!CHECK: acc.atomic.write %[[X_DECL]]#1 = %[[VAR_7y]] : !fir.ref<i32>, i32
!CHECK: return
!CHECK: }

program acc_atomic_write_test
    integer :: x, y

    !$acc atomic write
        x = 7 * y

end program acc_atomic_write_test

! Test lowering atomic read for pointer variables.

!CHECK-LABEL: func.func @_QPatomic_write_pointer() {
!CHECK: %[[X:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "x", uniq_name = "_QFatomic_write_pointerEx"}
!CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFatomic_write_pointerEx"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK: %[[C1:.*]] = arith.constant 1 : i32
!CHECK: %[[LOAD_X:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[BOX_ADDR_X:.*]] = fir.box_addr %[[LOAD_X]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: acc.atomic.write %[[BOX_ADDR_X]] = %[[C1]] : !fir.ptr<i32>, i32
!CHECK: %[[C2:.*]] = arith.constant 2 : i32
!CHECK: %[[LOAD_X:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[BOX_ADDR_X:.*]] = fir.box_addr %[[LOAD_X]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: hlfir.assign %[[C2]] to %[[BOX_ADDR_X]] : i32, !fir.ptr<i32>

subroutine atomic_write_pointer()
  integer, pointer :: x

  !$acc atomic write
    x = 1

  x = 2
end subroutine

!CHECK-LABEL: func.func @_QPatomic_write_typed_assign
!CHECK: %[[R2:.*]] = fir.alloca f32 {bindc_name = "r2", uniq_name = "{{.*}}r2"}
!CHECK: %[[R2_DECL:.*]]:2 = hlfir.declare %[[R2]] {uniq_name = "_QFatomic_write_typed_assignEr2"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
!CHECK: acc.atomic.write %[[R2_DECL]]#1 = %[[CST]]   : !fir.ref<f32>, f32

subroutine atomic_write_typed_assign
  real :: r2
  !$acc atomic write
  r2 = 0
end subroutine
