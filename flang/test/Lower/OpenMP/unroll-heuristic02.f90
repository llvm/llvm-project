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
!CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG5]] dummy_scope %[[VAL_0]] {uniq_name = "_QFomp_unroll_heuristic_nested02Einner_inc"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[ARG3]] dummy_scope %[[VAL_0]] {uniq_name = "_QFomp_unroll_heuristic_nested02Einner_lb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[ARG4]] dummy_scope %[[VAL_0]] {uniq_name = "_QFomp_unroll_heuristic_nested02Einner_ub"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFomp_unroll_heuristic_nested02Ej"}
!CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFomp_unroll_heuristic_nested02Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] {uniq_name = "_QFomp_unroll_heuristic_nested02Eouter_inc"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QFomp_unroll_heuristic_nested02Eouter_lb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QFomp_unroll_heuristic_nested02Eouter_ub"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_11:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_unroll_heuristic_nested02Eres"}
!CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] {uniq_name = "_QFomp_unroll_heuristic_nested02Eres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_13:.*]] = fir.alloca i32 {bindc_name = "i", pinned, uniq_name = "_QFomp_unroll_heuristic_nested02Ei"}
!CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]] {uniq_name = "_QFomp_unroll_heuristic_nested02Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_15:.*]] = fir.alloca i32 {bindc_name = "j", pinned, uniq_name = "_QFomp_unroll_heuristic_nested02Ej"}
!CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_15]] {uniq_name = "_QFomp_unroll_heuristic_nested02Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
!CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<i32>
!CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<i32>
!CHECK:           %[[VAL_20:.*]] = arith.constant 0 : i32
!CHECK:           %[[VAL_21:.*]] = arith.constant 1 : i32
!CHECK:           %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_19]], %[[VAL_20]] : i32
!CHECK:           %[[VAL_23:.*]] = arith.subi %[[VAL_20]], %[[VAL_19]] : i32
!CHECK:           %[[VAL_24:.*]] = arith.select %[[VAL_22]], %[[VAL_23]], %[[VAL_19]] : i32
!CHECK:           %[[VAL_25:.*]] = arith.select %[[VAL_22]], %[[VAL_18]], %[[VAL_17]] : i32
!CHECK:           %[[VAL_26:.*]] = arith.select %[[VAL_22]], %[[VAL_17]], %[[VAL_18]] : i32
!CHECK:           %[[VAL_27:.*]] = arith.subi %[[VAL_26]], %[[VAL_25]] overflow<nuw> : i32
!CHECK:           %[[VAL_28:.*]] = arith.divui %[[VAL_27]], %[[VAL_24]] : i32
!CHECK:           %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_21]] overflow<nuw> : i32
!CHECK:           %[[VAL_30:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_25]] : i32
!CHECK:           %[[VAL_31:.*]] = arith.select %[[VAL_30]], %[[VAL_20]], %[[VAL_29]] : i32
!CHECK:           %[[VAL_32:.*]] = omp.new_cli
!CHECK:           omp.canonical_loop(%[[VAL_32]]) %[[VAL_33:.*]] : i32 in range(%[[VAL_31]]) {
!CHECK:             %[[VAL_34:.*]] = arith.muli %[[VAL_33]], %[[VAL_19]] : i32
!CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_17]], %[[VAL_34]] : i32
!CHECK:             hlfir.assign %[[VAL_35]] to %[[VAL_14]]#0 : i32, !fir.ref<i32>
!CHECK:             %[[VAL_36:.*]] = fir.alloca i32 {bindc_name = "j", pinned, uniq_name = "_QFomp_unroll_heuristic_nested02Ej"}
!CHECK:             %[[VAL_37:.*]]:2 = hlfir.declare %[[VAL_36]] {uniq_name = "_QFomp_unroll_heuristic_nested02Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:             %[[VAL_38:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
!CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
!CHECK:             %[[VAL_40:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
!CHECK:             %[[VAL_41:.*]] = arith.constant 0 : i32
!CHECK:             %[[VAL_42:.*]] = arith.constant 1 : i32
!CHECK:             %[[VAL_43:.*]] = arith.cmpi slt, %[[VAL_40]], %[[VAL_41]] : i32
!CHECK:             %[[VAL_44:.*]] = arith.subi %[[VAL_41]], %[[VAL_40]] : i32
!CHECK:             %[[VAL_45:.*]] = arith.select %[[VAL_43]], %[[VAL_44]], %[[VAL_40]] : i32
!CHECK:             %[[VAL_46:.*]] = arith.select %[[VAL_43]], %[[VAL_39]], %[[VAL_38]] : i32
!CHECK:             %[[VAL_47:.*]] = arith.select %[[VAL_43]], %[[VAL_38]], %[[VAL_39]] : i32
!CHECK:             %[[VAL_48:.*]] = arith.subi %[[VAL_47]], %[[VAL_46]] overflow<nuw> : i32
!CHECK:             %[[VAL_49:.*]] = arith.divui %[[VAL_48]], %[[VAL_45]] : i32
!CHECK:             %[[VAL_50:.*]] = arith.addi %[[VAL_49]], %[[VAL_42]] overflow<nuw> : i32
!CHECK:             %[[VAL_51:.*]] = arith.cmpi slt, %[[VAL_47]], %[[VAL_46]] : i32
!CHECK:             %[[VAL_52:.*]] = arith.select %[[VAL_51]], %[[VAL_41]], %[[VAL_50]] : i32
!CHECK:             %[[VAL_53:.*]] = omp.new_cli
!CHECK:             omp.canonical_loop(%[[VAL_53]]) %[[VAL_54:.*]] : i32 in range(%[[VAL_52]]) {
!CHECK:               %[[VAL_55:.*]] = arith.muli %[[VAL_54]], %[[VAL_40]] : i32
!CHECK:               %[[VAL_56:.*]] = arith.addi %[[VAL_38]], %[[VAL_55]] : i32
!CHECK:               hlfir.assign %[[VAL_56]] to %[[VAL_37]]#0 : i32, !fir.ref<i32>
!CHECK:               %[[VAL_57:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<i32>
!CHECK:               %[[VAL_58:.*]] = fir.load %[[VAL_37]]#0 : !fir.ref<i32>
!CHECK:               %[[VAL_59:.*]] = arith.addi %[[VAL_57]], %[[VAL_58]] : i32
!CHECK:               hlfir.assign %[[VAL_59]] to %[[VAL_12]]#0 : i32, !fir.ref<i32>
!CHECK:               omp.terminator
!CHECK:             }
!CHECK:             omp.unroll_heuristic(%[[VAL_53]])
!CHECK:             omp.terminator
!CHECK:           }
!CHECK:           omp.unroll_heuristic(%[[VAL_32]])
!CHECK:           return
!CHECK:         }
