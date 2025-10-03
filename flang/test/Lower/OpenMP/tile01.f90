! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s | FileCheck %s


subroutine omp_tile01(lb, ub, inc)
  integer res, i, lb, ub, inc

  !$omp tile sizes(4)
  do i = lb, ub, inc
    res = i
  end do
  !$omp end tile

end subroutine omp_tile01


! CHECK: func.func @_QPomp_tile01(
! CHECK:    %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "lb"},
! CHECK:    %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "ub"},
! CHECK:    %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "inc"}) {
! CHECK:         %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_tile01Ei"}
! CHECK:         %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFomp_tile01Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] {uniq_name = "_QFomp_tile01Einc"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_4:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QFomp_tile01Elb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_tile01Eres"}
! CHECK:         %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFomp_tile01Eres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_7:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QFomp_tile01Eub"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_8:.*]] = arith.constant 4 : i32
! CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
! CHECK:         %[[VAL_11:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:         %[[VAL_12:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_13:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_12]] : i32
! CHECK:         %[[VAL_15:.*]] = arith.subi %[[VAL_12]], %[[VAL_11]] : i32
! CHECK:         %[[VAL_16:.*]] = arith.select %[[VAL_14]], %[[VAL_15]], %[[VAL_11]] : i32
! CHECK:         %[[VAL_17:.*]] = arith.select %[[VAL_14]], %[[VAL_10]], %[[VAL_9]] : i32
! CHECK:         %[[VAL_18:.*]] = arith.select %[[VAL_14]], %[[VAL_9]], %[[VAL_10]] : i32
! CHECK:         %[[VAL_19:.*]] = arith.subi %[[VAL_18]], %[[VAL_17]] overflow<nuw> : i32
! CHECK:         %[[VAL_20:.*]] = arith.divui %[[VAL_19]], %[[VAL_16]] : i32
! CHECK:         %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_13]] overflow<nuw> : i32
! CHECK:         %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_17]] : i32
! CHECK:         %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_12]], %[[VAL_21]] : i32
! CHECK:         %[[VAL_24:.*]] = omp.new_cli
! CHECK:         omp.canonical_loop(%[[VAL_24]]) %[[VAL_25:.*]] : i32 in range(%[[VAL_23]]) {
! CHECK:           %[[VAL_26:.*]] = arith.muli %[[VAL_25]], %[[VAL_11]] : i32
! CHECK:           %[[VAL_27:.*]] = arith.addi %[[VAL_9]], %[[VAL_26]] : i32
! CHECK:           hlfir.assign %[[VAL_27]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>
! CHECK:           %[[VAL_28:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:           hlfir.assign %[[VAL_28]] to %[[VAL_6]]#0 : i32, !fir.ref<i32>
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         %[[VAL_29:.*]] = omp.new_cli
! CHECK:         %[[VAL_30:.*]] = omp.new_cli
! CHECK:         omp.tile (%[[VAL_29]], %[[VAL_30]]) <- (%[[VAL_24]]) sizes(%[[VAL_8]] : i32)
! CHECK:         return
! CHECK:       }

