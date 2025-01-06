! RUN: %flang_fc1 -funsigned -emit-mlir %s -o - | FileCheck %s

unsigned function f01(u, v)
  unsigned, intent(in) :: u, v
  f01 = u + v - 1u
end

!CHECK: func.func @_QPf01(%[[ARG0:.*]]: !fir.ref<ui32> {fir.bindc_name = "u"}, %[[ARG1:.*]]: !fir.ref<ui32> {fir.bindc_name = "v"}) -> ui32 {
!CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
!CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
!CHECK: %[[VAL_1:.*]] = fir.alloca ui32 {bindc_name = "f01", uniq_name = "_QFf01Ef01"}
!CHECK: %[[VAL_2:.*]] = fir.declare %[[VAL_1]] {uniq_name = "_QFf01Ef01"} : (!fir.ref<ui32>) -> !fir.ref<ui32>
!CHECK: %[[VAL_3:.*]] = fir.declare %[[ARG0]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFf01Eu"} : (!fir.ref<ui32>, !fir.dscope) -> !fir.ref<ui32>
!CHECK: %[[VAL_4:.*]] = fir.declare %[[ARG1]] dummy_scope %[[VAL_0]] {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFf01Ev"} : (!fir.ref<ui32>, !fir.dscope) -> !fir.ref<ui32>
!CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_3]] : !fir.ref<ui32>
!CHECK: %[[VAL_6:.*]] = fir.load %[[VAL_4]] : !fir.ref<ui32>
!CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (ui32) -> i32
!CHECK: %[[VAL_8:.*]] = fir.convert %[[VAL_6]] : (ui32) -> i32
!CHECK: %[[VAL_9:.*]] = arith.addi %[[VAL_7]], %[[VAL_8]] : i32
!CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> ui32
!CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (ui32) -> i32
!CHECK: %[[VAL_12:.*]] = arith.subi %[[VAL_11]], %[[C1_I32]] : i32
!CHECK: %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> ui32
!CHECK: fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<ui32>
!CHECK: %[[VAL_14:.*]] = fir.load %[[VAL_2]] : !fir.ref<ui32>
!CHECK: return %[[VAL_14]] : ui32
