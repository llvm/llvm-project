! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s | FileCheck %s


subroutine omp_tile02(lb, ub, inc)
  integer res, i, lb, ub, inc

  !$omp tile sizes(3,7)
  do i = lb, ub, inc
    do j = lb, ub, inc
      res = i + j
    end do
  end do
  !$omp end tile

end subroutine omp_tile02


! CHECK: func.func @_QPomp_tile02(
! CHECK:    %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "lb"},
! CHECK:    %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "ub"},
! CHECK:    %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "inc"}) {
! CHECK:         %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_tile02Ei"}
! CHECK:         %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFomp_tile02Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] {uniq_name = "_QFomp_tile02Einc"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFomp_tile02Ej"}
! CHECK:         %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFomp_tile02Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_6:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {uniq_name = "_QFomp_tile02Elb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_7:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_tile02Eres"}
! CHECK:         %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] {uniq_name = "_QFomp_tile02Eres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_9:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {uniq_name = "_QFomp_tile02Eub"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_10:.*]] = arith.constant 3 : i32
! CHECK:         %[[VAL_11:.*]] = arith.constant 7 : i32
! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! CHECK:         %[[VAL_13:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
! CHECK:         %[[VAL_14:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:         %[[VAL_15:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_15]] : i32
! CHECK:         %[[VAL_18:.*]] = arith.subi %[[VAL_15]], %[[VAL_14]] : i32
! CHECK:         %[[VAL_19:.*]] = arith.select %[[VAL_17]], %[[VAL_18]], %[[VAL_14]] : i32
! CHECK:         %[[VAL_20:.*]] = arith.select %[[VAL_17]], %[[VAL_13]], %[[VAL_12]] : i32
! CHECK:         %[[VAL_21:.*]] = arith.select %[[VAL_17]], %[[VAL_12]], %[[VAL_13]] : i32
! CHECK:         %[[VAL_22:.*]] = arith.subi %[[VAL_21]], %[[VAL_20]] overflow<nuw> : i32
! CHECK:         %[[VAL_23:.*]] = arith.divui %[[VAL_22]], %[[VAL_19]] : i32
! CHECK:         %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_16]] overflow<nuw> : i32
! CHECK:         %[[VAL_25:.*]] = arith.cmpi slt, %[[VAL_21]], %[[VAL_20]] : i32
! CHECK:         %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_15]], %[[VAL_24]] : i32
! CHECK:         %[[VAL_27:.*]] = omp.new_cli
! CHECK:         %[[VAL_28:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<i32>
! CHECK:         %[[VAL_29:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
! CHECK:         %[[VAL_30:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:         %[[VAL_31:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_32:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_33:.*]] = arith.cmpi slt, %[[VAL_30]], %[[VAL_31]] : i32
! CHECK:         %[[VAL_34:.*]] = arith.subi %[[VAL_31]], %[[VAL_30]] : i32
! CHECK:         %[[VAL_35:.*]] = arith.select %[[VAL_33]], %[[VAL_34]], %[[VAL_30]] : i32
! CHECK:         %[[VAL_36:.*]] = arith.select %[[VAL_33]], %[[VAL_29]], %[[VAL_28]] : i32
! CHECK:         %[[VAL_37:.*]] = arith.select %[[VAL_33]], %[[VAL_28]], %[[VAL_29]] : i32
! CHECK:         %[[VAL_38:.*]] = arith.subi %[[VAL_37]], %[[VAL_36]] overflow<nuw> : i32
! CHECK:         %[[VAL_39:.*]] = arith.divui %[[VAL_38]], %[[VAL_35]] : i32
! CHECK:         %[[VAL_40:.*]] = arith.addi %[[VAL_39]], %[[VAL_32]] overflow<nuw> : i32
! CHECK:         %[[VAL_41:.*]] = arith.cmpi slt, %[[VAL_37]], %[[VAL_36]] : i32
! CHECK:         %[[VAL_42:.*]] = arith.select %[[VAL_41]], %[[VAL_31]], %[[VAL_40]] : i32
! CHECK:         %[[VAL_43:.*]] = omp.new_cli
! CHECK:         omp.canonical_loop(%[[VAL_27]]) %[[VAL_44:.*]] : i32 in range(%[[VAL_26]]) {
! CHECK:           omp.canonical_loop(%[[VAL_43]]) %[[VAL_45:.*]] : i32 in range(%[[VAL_42]]) {
! CHECK:             %[[VAL_46:.*]] = arith.muli %[[VAL_44]], %[[VAL_14]] : i32
! CHECK:             %[[VAL_47:.*]] = arith.addi %[[VAL_12]], %[[VAL_46]] : i32
! CHECK:             hlfir.assign %[[VAL_47]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>
! CHECK:             %[[VAL_48:.*]] = arith.muli %[[VAL_45]], %[[VAL_30]] : i32
! CHECK:             %[[VAL_49:.*]] = arith.addi %[[VAL_28]], %[[VAL_48]] : i32
! CHECK:             hlfir.assign %[[VAL_49]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
! CHECK:             %[[VAL_50:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_51:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_52:.*]] = arith.addi %[[VAL_50]], %[[VAL_51]] : i32
! CHECK:             hlfir.assign %[[VAL_52]] to %[[VAL_8]]#0 : i32, !fir.ref<i32>
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         %[[VAL_53:.*]] = omp.new_cli
! CHECK:         %[[VAL_54:.*]] = omp.new_cli
! CHECK:         %[[VAL_55:.*]] = omp.new_cli
! CHECK:         %[[VAL_56:.*]] = omp.new_cli
! CHECK:         omp.tile (%[[VAL_53]], %[[VAL_55]], %[[VAL_54]], %[[VAL_56]]) <- (%[[VAL_27]], %[[VAL_43]]) sizes(%[[VAL_10]], %[[VAL_11]] : i32, i32)
! CHECK:         return
! CHECK:       }
