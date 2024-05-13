// Test hlfir.concat operation lowering to operations operating on memory.

// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s


func.func @concat(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>, %arg2: !fir.boxchar<1>) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "c1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 {uniq_name = "c2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 {uniq_name = "c3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6 = arith.addi %2#1, %4#1 : index
  %7 = hlfir.concat %3#0, %5#0 len %6 : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %7 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  return
}
// CHECK-LABEL:   func.func @concat(
// CHECK-SAME:    %[[VAL_0:[^:]*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_1:[^:]*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_2:[^:]*]]: !fir.boxchar<1>) {
// CHECK:  %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]]#0 typeparams %[[VAL_3]]#1 {uniq_name = "c1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 {uniq_name = "c2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]]#0 typeparams %[[VAL_7]]#1 {uniq_name = "c3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  %[[VAL_9:.*]] = arith.addi %[[VAL_5]]#1, %[[VAL_7]]#1 : index
// CHECK:  %[[VAL_10:.*]] = arith.addi %[[VAL_5]]#1, %[[VAL_7]]#1 : index
// CHECK:  %[[VAL_11:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_10]] : index) {bindc_name = ".chrtmp"}
// CHECK:  %[[VAL_12:.*]] = arith.constant 1 : i64
// CHECK:  %[[VAL_13:.*]] = fir.convert %[[VAL_5]]#1 : (index) -> i64
// CHECK:  %[[VAL_14:.*]] = arith.muli %[[VAL_12]], %[[VAL_13]] : i64
// CHECK:  %[[VAL_15:.*]] = arith.constant false
// CHECK:  %[[VAL_16:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:  %[[VAL_17:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:  fir.call @llvm.memmove.p0.p0.i64(%[[VAL_16]], %[[VAL_17]], %[[VAL_14]], %[[VAL_15]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
// CHECK:  %[[VAL_18:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_19:.*]] = arith.subi %[[VAL_10]], %[[VAL_18]] : index
// CHECK:  fir.do_loop %[[VAL_20:.*]] = %[[VAL_5]]#1 to %[[VAL_19]] step %[[VAL_18]] {
// CHECK:    %[[VAL_21:.*]] = arith.subi %[[VAL_20]], %[[VAL_5]]#1 : index
// CHECK:    %[[VAL_22:.*]] = fir.convert %[[VAL_8]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:    %[[VAL_23:.*]] = fir.coordinate_of %[[VAL_22]], %[[VAL_21]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:    %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<!fir.char<1>>
// CHECK:    %[[VAL_25:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:    %[[VAL_26:.*]] = fir.coordinate_of %[[VAL_25]], %[[VAL_20]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:    fir.store %[[VAL_24]] to %[[VAL_26]] : !fir.ref<!fir.char<1>>
// CHECK:  }
// CHECK:  %[[VAL_27:.*]]:2 = hlfir.declare %[[VAL_11]] typeparams %[[VAL_10]] {uniq_name = "tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  hlfir.assign %[[VAL_27]]#0 to %[[VAL_4]]#0 : !fir.boxchar<1>, !fir.boxchar<1>


func.func @concat_chained(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>, %arg2: !fir.boxchar<1>, %arg3: !fir.boxchar<1>) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "c1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 {uniq_name = "c2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %4:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %5:2 = hlfir.declare %4#0 typeparams %4#1 {uniq_name = "c3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %6:2 = fir.unboxchar %arg3 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %7:2 = hlfir.declare %6#0 typeparams %6#1 {uniq_name = "c4"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %8 = arith.addi %2#1, %4#1 : index
  %9 = hlfir.concat %3#0, %5#0 len %8 : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  %10 = arith.addi %8, %6#1 : index
  %11 = hlfir.concat %9, %7#0 len %10 : (!hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %11 to %1#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  return
}
// CHECK-LABEL:   func.func @concat_chained(
// CHECK-SAME:    %[[VAL_0:[^:]*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_1:[^:]*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_2:[^:]*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_3:[^:]*]]: !fir.boxchar<1>) {
// CHECK:  %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 {uniq_name = "c1"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  %[[VAL_6:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]]#0 typeparams %[[VAL_6]]#1 {uniq_name = "c2"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  %[[VAL_8:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]]#0 typeparams %[[VAL_8]]#1 {uniq_name = "c3"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  %[[VAL_10:.*]]:2 = fir.unboxchar %[[VAL_3]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:  %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_10]]#0 typeparams %[[VAL_10]]#1 {uniq_name = "c4"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  %[[VAL_12:.*]] = arith.addi %[[VAL_6]]#1, %[[VAL_8]]#1 : index
// CHECK:  %[[VAL_13:.*]] = arith.addi %[[VAL_6]]#1, %[[VAL_8]]#1 : index
// CHECK:  %[[VAL_14:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_13]] : index) {bindc_name = ".chrtmp"}
// CHECK:  %[[VAL_15:.*]] = arith.constant 1 : i64
// CHECK:  %[[VAL_16:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
// CHECK:  %[[VAL_17:.*]] = arith.muli %[[VAL_15]], %[[VAL_16]] : i64
// CHECK:  %[[VAL_18:.*]] = arith.constant false
// CHECK:  %[[VAL_19:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:  %[[VAL_20:.*]] = fir.convert %[[VAL_7]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:  fir.call @llvm.memmove.p0.p0.i64(%[[VAL_19]], %[[VAL_20]], %[[VAL_17]], %[[VAL_18]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
// CHECK:  %[[VAL_21:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_22:.*]] = arith.subi %[[VAL_13]], %[[VAL_21]] : index
// CHECK:  fir.do_loop %[[VAL_23:.*]] = %[[VAL_6]]#1 to %[[VAL_22]] step %[[VAL_21]] {
// CHECK:    %[[VAL_24:.*]] = arith.subi %[[VAL_23]], %[[VAL_6]]#1 : index
// CHECK:    %[[VAL_25:.*]] = fir.convert %[[VAL_9]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:    %[[VAL_26:.*]] = fir.coordinate_of %[[VAL_25]], %[[VAL_24]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:    %[[VAL_27:.*]] = fir.load %[[VAL_26]] : !fir.ref<!fir.char<1>>
// CHECK:    %[[VAL_28:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:    %[[VAL_29:.*]] = fir.coordinate_of %[[VAL_28]], %[[VAL_23]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:    fir.store %[[VAL_27]] to %[[VAL_29]] : !fir.ref<!fir.char<1>>
// CHECK:  }
// CHECK:  %[[VAL_30:.*]]:2 = hlfir.declare %[[VAL_14]] typeparams %[[VAL_13]] {uniq_name = "tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  %[[VAL_31:.*]] = arith.addi %[[VAL_12]], %[[VAL_10]]#1 : index
// CHECK:  %[[VAL_32:.*]] = arith.addi %[[VAL_13]], %[[VAL_10]]#1 : index
// CHECK:  %[[VAL_33:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_32]] : index) {bindc_name = ".chrtmp"}
// CHECK:  %[[VAL_34:.*]] = arith.constant 1 : i64
// CHECK:  %[[VAL_35:.*]] = fir.convert %[[VAL_13]] : (index) -> i64
// CHECK:  %[[VAL_36:.*]] = arith.muli %[[VAL_34]], %[[VAL_35]] : i64
// CHECK:  %[[VAL_37:.*]] = arith.constant false
// CHECK:  %[[VAL_38:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:  %[[VAL_39:.*]] = fir.convert %[[VAL_30]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:  fir.call @llvm.memmove.p0.p0.i64(%[[VAL_38]], %[[VAL_39]], %[[VAL_36]], %[[VAL_37]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
// CHECK:  %[[VAL_40:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_41:.*]] = arith.subi %[[VAL_32]], %[[VAL_40]] : index
// CHECK:  fir.do_loop %[[VAL_42:.*]] = %[[VAL_13]] to %[[VAL_41]] step %[[VAL_40]] {
// CHECK:    %[[VAL_43:.*]] = arith.subi %[[VAL_42]], %[[VAL_13]] : index
// CHECK:    %[[VAL_44:.*]] = fir.convert %[[VAL_11]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:    %[[VAL_45:.*]] = fir.coordinate_of %[[VAL_44]], %[[VAL_43]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:    %[[VAL_46:.*]] = fir.load %[[VAL_45]] : !fir.ref<!fir.char<1>>
// CHECK:    %[[VAL_47:.*]] = fir.convert %[[VAL_33]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:    %[[VAL_48:.*]] = fir.coordinate_of %[[VAL_47]], %[[VAL_42]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:    fir.store %[[VAL_46]] to %[[VAL_48]] : !fir.ref<!fir.char<1>>
// CHECK:  }
// CHECK:  %[[VAL_49:.*]]:2 = hlfir.declare %[[VAL_33]] typeparams %[[VAL_32]] {uniq_name = "tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  hlfir.assign %[[VAL_49]]#0 to %[[VAL_5]]#0 : !fir.boxchar<1>, !fir.boxchar<1>

