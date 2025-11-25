! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

subroutine omp_unroll_heuristic_nested02(outer_lb, outer_ub, outer_inc, inner_lb, inner_ub, inner_inc)
  integer res, i, j, inner_lb, inner_ub, inner_inc, outer_lb, outer_ub, outer_inc

  !$omp unroll
  do i = outer_lb, outer_ub, outer_inc
    !$omp unroll
    do j = inner_lb, inner_ub, inner_inc
      res = i + j
    end do
    !$omp end unroll
  end do
  !$omp end unroll

end subroutine omp_unroll_heuristic_nested02


!CHECK-LABEL:   func.func @_QPomp_unroll_heuristic_nested02(
!CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "outer_lb"},
!CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "outer_ub"},
!CHECK-SAME:      %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "outer_inc"},
!CHECK-SAME:      %[[ARG3:.*]]: !fir.ref<i32> {fir.bindc_name = "inner_lb"},
!CHECK-SAME:      %[[ARG4:.*]]: !fir.ref<i32> {fir.bindc_name = "inner_ub"},
!CHECK-SAME:      %[[ARG5:.*]]: !fir.ref<i32> {fir.bindc_name = "inner_inc"}) {
!CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
!CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_unroll_heuristic_nested02Ei"}
!CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFomp_unroll_heuristic_nested02Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG5]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_unroll_heuristic_nested02Einner_inc"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[ARG3]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_unroll_heuristic_nested02Einner_lb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG4]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_unroll_heuristic_nested02Einner_ub"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFomp_unroll_heuristic_nested02Ej"}
!CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFomp_unroll_heuristic_nested02Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_unroll_heuristic_nested02Eouter_inc"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_unroll_heuristic_nested02Eouter_lb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} {uniq_name = "_QFomp_unroll_heuristic_nested02Eouter_ub"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_11:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_unroll_heuristic_nested02Eres"}
!CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] {uniq_name = "_QFomp_unroll_heuristic_nested02Eres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
!CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<i32>
!CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
!CHECK:           %[[VAL_16:.*]] = arith.constant 0 : i32
!CHECK:           %[[VAL_17:.*]] = arith.constant 1 : i32
!CHECK:           %[[VAL_18:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_16]] : i32
!CHECK:           %[[VAL_19:.*]] = arith.subi %[[VAL_16]], %[[VAL_15]] : i32
!CHECK:           %[[VAL_20:.*]] = arith.select %[[VAL_18]], %[[VAL_19]], %[[VAL_15]] : i32
!CHECK:           %[[VAL_21:.*]] = arith.select %[[VAL_18]], %[[VAL_14]], %[[VAL_13]] : i32
!CHECK:           %[[VAL_22:.*]] = arith.select %[[VAL_18]], %[[VAL_13]], %[[VAL_14]] : i32
!CHECK:           %[[VAL_23:.*]] = arith.subi %[[VAL_22]], %[[VAL_21]] overflow<nuw> : i32
!CHECK:           %[[VAL_24:.*]] = arith.divui %[[VAL_23]], %[[VAL_20]] : i32
!CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_17]] overflow<nuw> : i32
!CHECK:           %[[VAL_26:.*]] = arith.cmpi slt, %[[VAL_22]], %[[VAL_21]] : i32
!CHECK:           %[[VAL_27:.*]] = arith.select %[[VAL_26]], %[[VAL_16]], %[[VAL_25]] : i32
!CHECK:           %[[VAL_28:.*]] = omp.new_cli
!CHECK:           omp.canonical_loop(%[[VAL_28]]) %[[VAL_29:.*]] : i32 in range(%[[VAL_27]]) {
!CHECK:             %[[VAL_30:.*]] = arith.muli %[[VAL_29]], %[[VAL_15]] : i32
!CHECK:             %[[VAL_31:.*]] = arith.addi %[[VAL_13]], %[[VAL_30]] : i32
!CHECK:             hlfir.assign %[[VAL_31]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>
!CHECK:             %[[VAL_32:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
!CHECK:             %[[VAL_33:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
!CHECK:             %[[VAL_34:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
!CHECK:             %[[VAL_35:.*]] = arith.constant 0 : i32
!CHECK:             %[[VAL_36:.*]] = arith.constant 1 : i32
!CHECK:             %[[VAL_37:.*]] = arith.cmpi slt, %[[VAL_34]], %[[VAL_35]] : i32
!CHECK:             %[[VAL_38:.*]] = arith.subi %[[VAL_35]], %[[VAL_34]] : i32
!CHECK:             %[[VAL_39:.*]] = arith.select %[[VAL_37]], %[[VAL_38]], %[[VAL_34]] : i32
!CHECK:             %[[VAL_40:.*]] = arith.select %[[VAL_37]], %[[VAL_33]], %[[VAL_32]] : i32
!CHECK:             %[[VAL_41:.*]] = arith.select %[[VAL_37]], %[[VAL_32]], %[[VAL_33]] : i32
!CHECK:             %[[VAL_42:.*]] = arith.subi %[[VAL_41]], %[[VAL_40]] overflow<nuw> : i32
!CHECK:             %[[VAL_43:.*]] = arith.divui %[[VAL_42]], %[[VAL_39]] : i32
!CHECK:             %[[VAL_44:.*]] = arith.addi %[[VAL_43]], %[[VAL_36]] overflow<nuw> : i32
!CHECK:             %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_41]], %[[VAL_40]] : i32
!CHECK:             %[[VAL_46:.*]] = arith.select %[[VAL_45]], %[[VAL_35]], %[[VAL_44]] : i32
!CHECK:             %[[VAL_47:.*]] = omp.new_cli
!CHECK:             omp.canonical_loop(%[[VAL_47]]) %[[VAL_48:.*]] : i32 in range(%[[VAL_46]]) {
!CHECK:               %[[VAL_49:.*]] = arith.muli %[[VAL_48]], %[[VAL_34]] : i32
!CHECK:               %[[VAL_50:.*]] = arith.addi %[[VAL_32]], %[[VAL_49]] : i32
!CHECK:               hlfir.assign %[[VAL_50]] to %[[VAL_7]]#0 : i32, !fir.ref<i32>
!CHECK:               %[[VAL_51:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
!CHECK:               %[[VAL_52:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
!CHECK:               %[[VAL_53:.*]] = arith.addi %[[VAL_51]], %[[VAL_52]] : i32
!CHECK:               hlfir.assign %[[VAL_53]] to %[[VAL_12]]#0 : i32, !fir.ref<i32>
!CHECK:               omp.terminator
!CHECK:             }
!CHECK:             omp.unroll_heuristic(%[[VAL_47]])
!CHECK:             omp.terminator
!CHECK:           }
!CHECK:           omp.unroll_heuristic(%[[VAL_28]])
!CHECK:           return
!CHECK:         }
