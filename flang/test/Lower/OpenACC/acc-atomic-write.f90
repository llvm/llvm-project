! RUN: bbc -fopenacc -emit-fir -hlfir=false %s -o - | FileCheck %s

! This test checks the lowering of atomic write

!CHECK: func @_QQmain() attributes {fir.bindc_name = "acc_atomic_write_test"} {
!CHECK: %[[VAR_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
!CHECK: %[[VAR_Y:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFEy"}
!CHECK: %[[CONST_7:.*]] = arith.constant 7 : i32
!CHECK: {{.*}} = fir.load %[[VAR_Y]] : !fir.ref<i32>
!CHECK: %[[VAR_7y:.*]] = arith.muli %[[CONST_7]], {{.*}} : i32
!CHECK: acc.atomic.write %[[VAR_X]] = %[[VAR_7y]] : !fir.ref<i32>, i32
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
!CHECK: %[[C1:.*]] = arith.constant 1 : i32
!CHECK: %[[LOAD_X:.*]] = fir.load %[[X]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[BOX_ADDR_X:.*]] = fir.box_addr %[[LOAD_X]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: acc.atomic.write %[[BOX_ADDR_X]] = %[[C1]] : !fir.ptr<i32>, i32
!CHECK: %[[C2:.*]] = arith.constant 2 : i32
!CHECK: %[[LOAD_X:.*]] = fir.load %[[X]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK: %[[BOX_ADDR_X:.*]] = fir.box_addr %[[LOAD_X]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK: fir.store %[[C2]] to %[[BOX_ADDR_X]] : !fir.ptr<i32>

subroutine atomic_write_pointer()
  integer, pointer :: x

  !$acc atomic write
    x = 1

  x = 2
end subroutine

!CHECK-LABEL: func.func @_QPatomic_write_typed_assign
!CHECK: %[[VAR:.*]] = fir.alloca f32 {bindc_name = "r2", uniq_name = "{{.*}}r2"}
!CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
!CHECK: acc.atomic.write %[[VAR]] = %[[CST]]   : !fir.ref<f32>, f32

subroutine atomic_write_typed_assign
  real :: r2
  !$acc atomic write
  r2 = 0
end subroutine
