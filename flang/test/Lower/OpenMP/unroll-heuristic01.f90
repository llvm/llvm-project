! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s


subroutine omp_unroll_heuristic01(lb, ub, inc)
  integer res, i, lb, ub, inc

  !$omp unroll
  do i = lb, ub, inc
    res = i
  end do
  !$omp end unroll

end subroutine omp_unroll_heuristic01


! CHECK-LABEL:   func.func @_QPomp_unroll_heuristic01(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "lb"},
! CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "ub"},
! CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "inc"}) {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_unroll_heuristic01Ei"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFomp_unroll_heuristic01Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_unroll_heuristic01Einc"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_unroll_heuristic01Elb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_unroll_heuristic01Eres"}
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFomp_unroll_heuristic01Eres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_unroll_heuristic01Eub"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_11:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_11]] : i32
! CHECK:           %[[VAL_14:.*]] = arith.subi %[[VAL_11]], %[[VAL_10]] : i32
! CHECK:           %[[VAL_15:.*]] = arith.select %[[VAL_13]], %[[VAL_14]], %[[VAL_10]] : i32
! CHECK:           %[[VAL_16:.*]] = arith.select %[[VAL_13]], %[[VAL_9]], %[[VAL_8]] : i32
! CHECK:           %[[VAL_17:.*]] = arith.select %[[VAL_13]], %[[VAL_8]], %[[VAL_9]] : i32
! CHECK:           %[[VAL_18:.*]] = arith.subi %[[VAL_17]], %[[VAL_16]] overflow<nuw> : i32
! CHECK:           %[[VAL_19:.*]] = arith.divui %[[VAL_18]], %[[VAL_15]] : i32
! CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_12]] overflow<nuw> : i32
! CHECK:           %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_17]], %[[VAL_16]] : i32
! CHECK:           %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_11]], %[[VAL_20]] : i32
! CHECK:           %[[VAL_23:.*]] = omp.new_cli
! CHECK:           omp.canonical_loop(%[[VAL_23]]) %[[VAL_24:.*]] : i32 in range(%[[VAL_22]]) {
! CHECK:             %[[VAL_25:.*]] = arith.muli %[[VAL_24]], %[[VAL_10]] : i32
! CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_8]], %[[VAL_25]] : i32
! CHECK:             hlfir.assign %[[VAL_26]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>
! CHECK:             %[[VAL_27:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:             hlfir.assign %[[VAL_27]] to %[[VAL_6]]#0 : i32, !fir.ref<i32>
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           omp.unroll_heuristic(%[[VAL_23]])
! CHECK:           return
! CHECK:         }