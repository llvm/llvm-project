! Test LASTPRIVATE with iteration variable.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: func @_QPlastprivate_iv_inc
!CHECK:    %[[I_MEM:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[I:.*]]:2 = hlfir.declare %[[I_MEM]] {uniq_name = "_QFlastprivate_iv_incEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[I2_MEM:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFlastprivate_iv_incEi"}
!CHECK:    %[[I2:.*]]:2 = hlfir.declare %[[I2_MEM]] {uniq_name = "_QFlastprivate_iv_incEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[LB:.*]] = arith.constant 4 : i32
!CHECK:    %[[UB:.*]] = arith.constant 10 : i32
!CHECK:    %[[STEP:.*]]  = arith.constant 3 : i32
!CHECK:    omp.wsloop for  (%[[IV:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
!CHECK:      fir.store %[[IV]] to %[[I]]#1 : !fir.ref<i32>
!CHECK:      %[[V:.*]] = arith.addi %[[IV]], %[[STEP]] : i32
!CHECK:      %[[C0:.*]] = arith.constant 0 : i32
!CHECK:      %[[STEP_NEG:.*]] = arith.cmpi slt, %[[STEP]], %[[C0]] : i32
!CHECK:      %[[V_LT:.*]] = arith.cmpi slt, %[[V]], %[[UB]] : i32
!CHECK:      %[[V_GT:.*]] = arith.cmpi sgt, %[[V]], %[[UB]] : i32
!CHECK:      %[[CMP:.*]] = arith.select %[[STEP_NEG]], %[[V_LT]], %[[V_GT]] : i1
!CHECK:      fir.if %[[CMP]] {
!CHECK:        fir.store %[[V]] to %[[I]]#1 : !fir.ref<i32>
!CHECK:        %[[I_VAL:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
!CHECK:        hlfir.assign %[[I_VAL]] to %[[I2]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:      }
!CHECK:      omp.yield
!CHECK:    }
subroutine lastprivate_iv_inc()
  integer :: i

  !$omp do lastprivate(i)
  do i = 4, 10, 3
  end do
  !$omp end do
end subroutine

!CHECK-LABEL: func @_QPlastprivate_iv_dec
!CHECK:    %[[I_MEM:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
!CHECK:    %[[I:.*]]:2 = hlfir.declare %[[I_MEM]] {uniq_name = "_QFlastprivate_iv_decEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[I2_MEM:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFlastprivate_iv_decEi"}
!CHECK:    %[[I2:.*]]:2 = hlfir.declare %[[I2_MEM]] {uniq_name = "_QFlastprivate_iv_decEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[LB:.*]] = arith.constant 10 : i32
!CHECK:    %[[UB:.*]] = arith.constant 1 : i32
!CHECK:    %[[STEP:.*]]  = arith.constant -3 : i32
!CHECK:    omp.wsloop for  (%[[IV:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
!CHECK:      fir.store %[[IV]] to %[[I]]#1 : !fir.ref<i32>
!CHECK:      %[[V:.*]] = arith.addi %[[IV]], %[[STEP]] : i32
!CHECK:      %[[C0:.*]] = arith.constant 0 : i32
!CHECK:      %[[STEP_NEG:.*]] = arith.cmpi slt, %[[STEP]], %[[C0]] : i32
!CHECK:      %[[V_LT:.*]] = arith.cmpi slt, %[[V]], %[[UB]] : i32
!CHECK:      %[[V_GT:.*]] = arith.cmpi sgt, %[[V]], %[[UB]] : i32
!CHECK:      %[[CMP:.*]] = arith.select %[[STEP_NEG]], %[[V_LT]], %[[V_GT]] : i1
!CHECK:      fir.if %[[CMP]] {
!CHECK:        fir.store %[[V]] to %[[I]]#1 : !fir.ref<i32>
!CHECK:        %[[I_VAL:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
!CHECK:        hlfir.assign %[[I_VAL]] to %[[I2]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:      }
!CHECK:      omp.yield
!CHECK:    }
subroutine lastprivate_iv_dec()
  integer :: i

  !$omp do lastprivate(i)
  do i = 10, 1, -3
  end do
  !$omp end do
end subroutine
