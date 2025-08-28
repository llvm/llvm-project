// Test hlfir.cmpchar operation lowering to a fir runtime call
// RUN: fir-opt %s -lower-hlfir-intrinsics | FileCheck %s

// HLFIR for the test below has been produced from reduced flang/test/Lower/Intrinsics/lge_lgt_lle_llt.f90
func.func @_QPlge_test() {
// CHECK-LABEL:   func.func @_QPlge_test() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 7 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.array<3x!fir.char<1,3>> {bindc_name = "c1", uniq_name = "_QFlge_testEc1"}
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_5]]) typeparams %[[VAL_2]] {uniq_name = "_QFlge_testEc1"} : (!fir.ref<!fir.array<3x!fir.char<1,3>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<3x!fir.char<1,3>>>, !fir.ref<!fir.array<3x!fir.char<1,3>>>)
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.array<3x!fir.char<1,7>> {bindc_name = "c2", uniq_name = "_QFlge_testEc2"}
// CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_8]]) typeparams %[[VAL_1]] {uniq_name = "_QFlge_testEc2"} : (!fir.ref<!fir.array<3x!fir.char<1,7>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<3x!fir.char<1,7>>>, !fir.ref<!fir.array<3x!fir.char<1,7>>>)
// CHECK:           %[[VAL_10:.*]] = fir.alloca !fir.array<3x!fir.logical<4>> {bindc_name = "l", uniq_name = "_QFlge_testEl"}
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_11]]) {uniq_name = "_QFlge_testEl"} : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.ref<!fir.array<3x!fir.logical<4>>>)
// CHECK:           %[[VAL_13:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.logical<4>> {
    %0 = fir.dummy_scope : !fir.dscope
    %c3 = arith.constant 3 : index
    %c3_0 = arith.constant 3 : index
    %1 = fir.alloca !fir.array<3x!fir.char<1,3>> {bindc_name = "c1", uniq_name = "_QFlge_testEc1"}
    %2 = fir.shape %c3_0 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %1(%2) typeparams %c3 {uniq_name = "_QFlge_testEc1"} : (!fir.ref<!fir.array<3x!fir.char<1,3>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<3x!fir.char<1,3>>>, !fir.ref<!fir.array<3x!fir.char<1,3>>>)
    %c7 = arith.constant 7 : index
    %c3_1 = arith.constant 3 : index
    %4 = fir.alloca !fir.array<3x!fir.char<1,7>> {bindc_name = "c2", uniq_name = "_QFlge_testEc2"}
    %5 = fir.shape %c3_1 : (index) -> !fir.shape<1>
    %6:2 = hlfir.declare %4(%5) typeparams %c7 {uniq_name = "_QFlge_testEc2"} : (!fir.ref<!fir.array<3x!fir.char<1,7>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<3x!fir.char<1,7>>>, !fir.ref<!fir.array<3x!fir.char<1,7>>>)
    %c3_2 = arith.constant 3 : index
    %7 = fir.alloca !fir.array<3x!fir.logical<4>> {bindc_name = "l", uniq_name = "_QFlge_testEl"}
    %8 = fir.shape %c3_2 : (index) -> !fir.shape<1>
    %9:2 = hlfir.declare %7(%8) {uniq_name = "_QFlge_testEl"} : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.ref<!fir.array<3x!fir.logical<4>>>)
    %10 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.logical<4>> {
    ^bb0(%arg0: index):
      %14 = hlfir.designate %3#0 (%arg0)  typeparams %c3 : (!fir.ref<!fir.array<3x!fir.char<1,3>>>, index, index) -> !fir.ref<!fir.char<1,3>>
      %15 = hlfir.designate %6#0 (%arg0)  typeparams %c7 : (!fir.ref<!fir.array<3x!fir.char<1,7>>>, index, index) -> !fir.ref<!fir.char<1,7>>
      %16 = hlfir.cmpchar sge %14 %15 : (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,7>>) -> i1
      %17 = fir.convert %16 : (i1) -> !fir.logical<4>
      hlfir.yield_element %17 : !fir.logical<4>
    }
    hlfir.assign %10 to %9#0 : !hlfir.expr<3x!fir.logical<4>>, !fir.ref<!fir.array<3x!fir.logical<4>>>
    hlfir.destroy %10 : !hlfir.expr<3x!fir.logical<4>>
// CHECK:           ^bb0(%[[VAL_14:.*]]: index):
// CHECK:             %[[VAL_15:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_14]])  typeparams %[[VAL_2]] : (!fir.ref<!fir.array<3x!fir.char<1,3>>>, index, index) -> !fir.ref<!fir.char<1,3>>
// CHECK:             %[[VAL_16:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_14]])  typeparams %[[VAL_1]] : (!fir.ref<!fir.array<3x!fir.char<1,7>>>, index, index) -> !fir.ref<!fir.char<1,7>>
// CHECK:             %[[VAL_17:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_2]] : (index) -> i64
// CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_1]] : (index) -> i64
// CHECK:             %[[VAL_21:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %[[VAL_20]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:             %[[VAL_22:.*]] = arith.cmpi sge, %[[VAL_21]], %[[VAL_0]] : i32
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i1) -> !fir.logical<4>
// CHECK:             hlfir.yield_element %[[VAL_23]] : !fir.logical<4>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_13]] to %[[VAL_12]]#0 : !hlfir.expr<3x!fir.logical<4>>, !fir.ref<!fir.array<3x!fir.logical<4>>>
// CHECK:           hlfir.destroy %[[VAL_13]] : !hlfir.expr<3x!fir.logical<4>>
    %11 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.logical<4>> {
    ^bb0(%arg0: index):
      %14 = hlfir.designate %3#0 (%arg0)  typeparams %c3 : (!fir.ref<!fir.array<3x!fir.char<1,3>>>, index, index) -> !fir.ref<!fir.char<1,3>>
      %15 = hlfir.designate %6#0 (%arg0)  typeparams %c7 : (!fir.ref<!fir.array<3x!fir.char<1,7>>>, index, index) -> !fir.ref<!fir.char<1,7>>
      %16 = hlfir.cmpchar sgt %14 %15 : (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,7>>) -> i1
      %17 = fir.convert %16 : (i1) -> !fir.logical<4>
      hlfir.yield_element %17 : !fir.logical<4>
    }
    hlfir.assign %11 to %9#0 : !hlfir.expr<3x!fir.logical<4>>, !fir.ref<!fir.array<3x!fir.logical<4>>>
    hlfir.destroy %11 : !hlfir.expr<3x!fir.logical<4>>
// CHECK:           %[[VAL_24:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.logical<4>> {
// CHECK:           ^bb0(%[[VAL_25:.*]]: index):
// CHECK:             %[[VAL_26:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_25]])  typeparams %[[VAL_2]] : (!fir.ref<!fir.array<3x!fir.char<1,3>>>, index, index) -> !fir.ref<!fir.char<1,3>>
// CHECK:             %[[VAL_27:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_25]])  typeparams %[[VAL_1]] : (!fir.ref<!fir.array<3x!fir.char<1,7>>>, index, index) -> !fir.ref<!fir.char<1,7>>
// CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_27]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_30:.*]] = fir.convert %[[VAL_2]] : (index) -> i64
// CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_1]] : (index) -> i64
// CHECK:             %[[VAL_32:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_28]], %[[VAL_29]], %[[VAL_30]], %[[VAL_31]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:             %[[VAL_33:.*]] = arith.cmpi sgt, %[[VAL_32]], %[[VAL_0]] : i32
// CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i1) -> !fir.logical<4>
// CHECK:             hlfir.yield_element %[[VAL_34]] : !fir.logical<4>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_24]] to %[[VAL_12]]#0 : !hlfir.expr<3x!fir.logical<4>>, !fir.ref<!fir.array<3x!fir.logical<4>>>
// CHECK:           hlfir.destroy %[[VAL_24]] : !hlfir.expr<3x!fir.logical<4>>
    %12 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.logical<4>> {
    ^bb0(%arg0: index):
      %14 = hlfir.designate %3#0 (%arg0)  typeparams %c3 : (!fir.ref<!fir.array<3x!fir.char<1,3>>>, index, index) -> !fir.ref<!fir.char<1,3>>
      %15 = hlfir.designate %6#0 (%arg0)  typeparams %c7 : (!fir.ref<!fir.array<3x!fir.char<1,7>>>, index, index) -> !fir.ref<!fir.char<1,7>>
      %16 = hlfir.cmpchar sle %14 %15 : (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,7>>) -> i1
      %17 = fir.convert %16 : (i1) -> !fir.logical<4>
      hlfir.yield_element %17 : !fir.logical<4>
    }
    hlfir.assign %12 to %9#0 : !hlfir.expr<3x!fir.logical<4>>, !fir.ref<!fir.array<3x!fir.logical<4>>>
    hlfir.destroy %12 : !hlfir.expr<3x!fir.logical<4>>
// CHECK:           %[[VAL_35:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.logical<4>> {
// CHECK:           ^bb0(%[[VAL_36:.*]]: index):
// CHECK:             %[[VAL_37:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_36]])  typeparams %[[VAL_2]] : (!fir.ref<!fir.array<3x!fir.char<1,3>>>, index, index) -> !fir.ref<!fir.char<1,3>>
// CHECK:             %[[VAL_38:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_36]])  typeparams %[[VAL_1]] : (!fir.ref<!fir.array<3x!fir.char<1,7>>>, index, index) -> !fir.ref<!fir.char<1,7>>
// CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_37]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_40:.*]] = fir.convert %[[VAL_38]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_2]] : (index) -> i64
// CHECK:             %[[VAL_42:.*]] = fir.convert %[[VAL_1]] : (index) -> i64
// CHECK:             %[[VAL_43:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_39]], %[[VAL_40]], %[[VAL_41]], %[[VAL_42]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:             %[[VAL_44:.*]] = arith.cmpi sle, %[[VAL_43]], %[[VAL_0]] : i32
// CHECK:             %[[VAL_45:.*]] = fir.convert %[[VAL_44]] : (i1) -> !fir.logical<4>
// CHECK:             hlfir.yield_element %[[VAL_45]] : !fir.logical<4>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_35]] to %[[VAL_12]]#0 : !hlfir.expr<3x!fir.logical<4>>, !fir.ref<!fir.array<3x!fir.logical<4>>>
// CHECK:           hlfir.destroy %[[VAL_35]] : !hlfir.expr<3x!fir.logical<4>>
    %13 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.logical<4>> {
    ^bb0(%arg0: index):
      %14 = hlfir.designate %3#0 (%arg0)  typeparams %c3 : (!fir.ref<!fir.array<3x!fir.char<1,3>>>, index, index) -> !fir.ref<!fir.char<1,3>>
      %15 = hlfir.designate %6#0 (%arg0)  typeparams %c7 : (!fir.ref<!fir.array<3x!fir.char<1,7>>>, index, index) -> !fir.ref<!fir.char<1,7>>
      %16 = hlfir.cmpchar slt %14 %15 : (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,7>>) -> i1
      %17 = fir.convert %16 : (i1) -> !fir.logical<4>
      hlfir.yield_element %17 : !fir.logical<4>
    }
    hlfir.assign %13 to %9#0 : !hlfir.expr<3x!fir.logical<4>>, !fir.ref<!fir.array<3x!fir.logical<4>>>
    hlfir.destroy %13 : !hlfir.expr<3x!fir.logical<4>>
    return
}
// CHECK:           %[[VAL_46:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.logical<4>> {
// CHECK:           ^bb0(%[[VAL_47:.*]]: index):
// CHECK:             %[[VAL_48:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_47]])  typeparams %[[VAL_2]] : (!fir.ref<!fir.array<3x!fir.char<1,3>>>, index, index) -> !fir.ref<!fir.char<1,3>>
// CHECK:             %[[VAL_49:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_47]])  typeparams %[[VAL_1]] : (!fir.ref<!fir.array<3x!fir.char<1,7>>>, index, index) -> !fir.ref<!fir.char<1,7>>
// CHECK:             %[[VAL_50:.*]] = fir.convert %[[VAL_48]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_51:.*]] = fir.convert %[[VAL_49]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<i8>
// CHECK:             %[[VAL_52:.*]] = fir.convert %[[VAL_2]] : (index) -> i64
// CHECK:             %[[VAL_53:.*]] = fir.convert %[[VAL_1]] : (index) -> i64
// CHECK:             %[[VAL_54:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_50]], %[[VAL_51]], %[[VAL_52]], %[[VAL_53]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:             %[[VAL_55:.*]] = arith.cmpi slt, %[[VAL_54]], %[[VAL_0]] : i32
// CHECK:             %[[VAL_56:.*]] = fir.convert %[[VAL_55]] : (i1) -> !fir.logical<4>
// CHECK:             hlfir.yield_element %[[VAL_56]] : !fir.logical<4>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_46]] to %[[VAL_12]]#0 : !hlfir.expr<3x!fir.logical<4>>, !fir.ref<!fir.array<3x!fir.logical<4>>>
// CHECK:           hlfir.destroy %[[VAL_46]] : !hlfir.expr<3x!fir.logical<4>>
// CHECK:           return
// CHECK:         }


// HLFIR for the test below has been produced
// from test case in flang/test/Lower/HLFIR/binary-ops.f90
// cmp_char2/cmp_char4 are produced from the modified original test to cover other character kinds.
func.func @_QPcmp_char(%arg0: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "x"}, %arg2: !fir.boxchar<1> {fir.bindc_name = "y"}) {
// CHECK-LABEL:   func.func @_QPcmp_char(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<1> {fir.bindc_name = "x"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.boxchar<1> {fir.bindc_name = "y"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFcmp_charEl"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]]#0 typeparams %[[VAL_3]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QFcmp_charEx"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QFcmp_charEy"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_4]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_3]]#1 : (index) -> i64
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
// CHECK:           %[[VAL_11:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
// CHECK:           %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_11]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i1) -> !fir.logical<4>
// CHECK:           hlfir.assign %[[VAL_13]] to %[[VAL_2]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK:           return
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFcmp_charEl"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %3:2 = hlfir.declare %2#0 typeparams %2#1 dummy_scope %0 {uniq_name = "_QFcmp_charEx"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %5:2 = hlfir.declare %4#0 typeparams %4#1 dummy_scope %0 {uniq_name = "_QFcmp_charEy"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %6 = hlfir.cmpchar eq %3#0 %5#0 : (!fir.boxchar<1>, !fir.boxchar<1>) -> i1
    %7 = fir.convert %6 : (i1) -> !fir.logical<4>
    hlfir.assign %7 to %1#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
    return
}

func.func @_QPcmp_char4(%arg0: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"}, %arg1: !fir.boxchar<4> {fir.bindc_name = "x"}, %arg2: !fir.boxchar<4> {fir.bindc_name = "y"}) {
// CHECK-LABEL:   func.func @_QPcmp_char4(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<4> {fir.bindc_name = "x"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.boxchar<4> {fir.bindc_name = "y"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFcmp_char4El"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]]#0 typeparams %[[VAL_3]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QFcmp_char4Ex"} : (!fir.ref<!fir.char<4,?>>, index, !fir.dscope) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QFcmp_char4Ey"} : (!fir.ref<!fir.char<4,?>>, index, !fir.dscope) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_4]]#1 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.ref<!fir.char<4,?>>) -> !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_3]]#1 : (index) -> i64
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
// CHECK:           %[[VAL_11:.*]] = fir.call @_FortranACharacterCompareScalar4(%[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]]) : (!fir.ref<i32>, !fir.ref<i32>, i64, i64) -> i32
// CHECK:           %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_11]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i1) -> !fir.logical<4>
// CHECK:           hlfir.assign %[[VAL_13]] to %[[VAL_2]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK:           return
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFcmp_char4El"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
    %3:2 = hlfir.declare %2#0 typeparams %2#1 dummy_scope %0 {uniq_name = "_QFcmp_char4Ex"} : (!fir.ref<!fir.char<4,?>>, index, !fir.dscope) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
    %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<4>) -> (!fir.ref<!fir.char<4,?>>, index)
    %5:2 = hlfir.declare %4#0 typeparams %4#1 dummy_scope %0 {uniq_name = "_QFcmp_char4Ey"} : (!fir.ref<!fir.char<4,?>>, index, !fir.dscope) -> (!fir.boxchar<4>, !fir.ref<!fir.char<4,?>>)
    %6 = hlfir.cmpchar eq %3#0 %5#0 : (!fir.boxchar<4>, !fir.boxchar<4>) -> i1
    %7 = fir.convert %6 : (i1) -> !fir.logical<4>
    hlfir.assign %7 to %1#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
    return
}

func.func @_QPcmp_char2(%arg0: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"}, %arg1: !fir.boxchar<2> {fir.bindc_name = "x"}, %arg2: !fir.boxchar<2> {fir.bindc_name = "y"}) {
// CHECK-LABEL:   func.func @_QPcmp_char2(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "l"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.boxchar<2> {fir.bindc_name = "x"},
// CHECK-SAME:      %[[ARG2:.*]]: !fir.boxchar<2> {fir.bindc_name = "y"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFcmp_char2El"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[ARG1]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]]#0 typeparams %[[VAL_3]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QFcmp_char2Ex"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[ARG2]] : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 dummy_scope %[[VAL_1]] {uniq_name = "_QFcmp_char2Ey"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
// CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_4]]#1 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<i16>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.ref<!fir.char<2,?>>) -> !fir.ref<i16>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_3]]#1 : (index) -> i64
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
// CHECK:           %[[VAL_11:.*]] = fir.call @_FortranACharacterCompareScalar2(%[[VAL_7]], %[[VAL_8]], %[[VAL_9]], %[[VAL_10]]) : (!fir.ref<i16>, !fir.ref<i16>, i64, i64) -> i32
// CHECK:           %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_11]], %[[VAL_0]] : i32
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i1) -> !fir.logical<4>
// CHECK:           hlfir.assign %[[VAL_13]] to %[[VAL_2]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK:           return
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFcmp_char2El"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
    %3:2 = hlfir.declare %2#0 typeparams %2#1 dummy_scope %0 {uniq_name = "_QFcmp_char2Ex"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
    %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<2>) -> (!fir.ref<!fir.char<2,?>>, index)
    %5:2 = hlfir.declare %4#0 typeparams %4#1 dummy_scope %0 {uniq_name = "_QFcmp_char2Ey"} : (!fir.ref<!fir.char<2,?>>, index, !fir.dscope) -> (!fir.boxchar<2>, !fir.ref<!fir.char<2,?>>)
    %6 = hlfir.cmpchar eq %3#0 %5#0 : (!fir.boxchar<2>, !fir.boxchar<2>) -> i1
    %7 = fir.convert %6 : (i1) -> !fir.logical<4>
    hlfir.assign %7 to %1#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
    return
}

