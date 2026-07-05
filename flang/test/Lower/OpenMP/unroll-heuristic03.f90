! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! Test implicitly privatized loop variable that is affected by unrolling.

subroutine omp_unroll_heuristic03(lb, ub, inc)
  integer res, i, lb, ub, inc

  !$omp parallel
    !$omp unroll
    do i = lb, ub, inc
      res = i
    end do
    !$omp end unroll
  !$omp end parallel

end subroutine omp_unroll_heuristic03


! CHECK-LABEL:   func.func @_QPomp_unroll_heuristic03(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "lb"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "ub"},
! CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "inc"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_unroll_heuristic03Ei"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFomp_unroll_heuristic03Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_unroll_heuristic03Einc"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_unroll_heuristic03Elb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_unroll_heuristic03Eres"}
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFomp_unroll_heuristic03Eres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_unroll_heuristic03Eub"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           omp.parallel private(@_QFomp_unroll_heuristic03Ei_private_i32 %[[VAL_2]]#0 -> %[[VAL_8:.*]] : !fir.ref<i32>) {
! CHECK:             %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFomp_unroll_heuristic03Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             %[[VAL_10:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_12:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_13:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_14:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_13]] : i32
! CHECK:             %[[VAL_16:.*]] = arith.subi %[[VAL_13]], %[[VAL_12]] : i32
! CHECK:             %[[VAL_17:.*]] = arith.select %[[VAL_15]], %[[VAL_16]], %[[VAL_12]] : i32
! CHECK:             %[[VAL_18:.*]] = arith.select %[[VAL_15]], %[[VAL_11]], %[[VAL_10]] : i32
! CHECK:             %[[VAL_19:.*]] = arith.select %[[VAL_15]], %[[VAL_10]], %[[VAL_11]] : i32
! CHECK:             %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_18]] overflow<nuw> : i32
! CHECK:             %[[VAL_21:.*]] = arith.divui %[[VAL_20]], %[[VAL_17]] : i32
! CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_14]] overflow<nuw> : i32
! CHECK:             %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_19]], %[[VAL_18]] : i32
! CHECK:             %[[VAL_24:.*]] = arith.select %[[VAL_23]], %[[VAL_13]], %[[VAL_22]] : i32
! CHECK:             %[[VAL_25:.*]] = omp.new_cli
! CHECK:             omp.canonical_loop(%[[VAL_25]]) %[[VAL_26:.*]] : i32 in range(%[[VAL_24]]) {
! CHECK:               %[[VAL_27:.*]] = arith.muli %[[VAL_26]], %[[VAL_12]] : i32
! CHECK:               %[[VAL_28:.*]] = arith.addi %[[VAL_10]], %[[VAL_27]] : i32
! CHECK:               hlfir.assign %[[VAL_28]] to %[[VAL_9]]#0 : i32, !fir.ref<i32>
! CHECK:               %[[VAL_29:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
! CHECK:               hlfir.assign %[[VAL_29]] to %[[VAL_6]]#0 : i32, !fir.ref<i32>
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.unroll_heuristic(%[[VAL_25]])
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }