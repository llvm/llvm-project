// Test hlfir.apply code generation
// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s

func.func @numeric_apply(%arg0 : !fir.ref<!fir.array<100xi32>>) {
  %expr = hlfir.as_expr %arg0 : (!fir.ref<!fir.array<100xi32>>) -> !hlfir.expr<100xi32>
  %c42 = arith.constant 42 : index
  %elem = hlfir.apply %expr, %c42 : (!hlfir.expr<100xi32>, index) -> i32
  return
}
// CHECK-LABEL:   func.func @numeric_apply(
// CHECK:    %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = ".tmp"}
// CHECK:    %[[VAL_9:.*]] = arith.constant 42 : index
// CHECK:    %[[VAL_10:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_9]])  : (!fir.heap<!fir.array<100xi32>>, index) -> !fir.ref<i32>
// CHECK:    %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>

func.func @character_apply(%arg0 : !fir.ref<!fir.array<?x!fir.char<2,?>>>, %l: index, %n: index) {
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  %c:2 = hlfir.declare %arg0(%shape) typeparams %l {uniq_name = "c"} : (!fir.ref<!fir.array<?x!fir.char<2,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<?x!fir.char<2,?>>>, !fir.ref<!fir.array<?x!fir.char<2,?>>>)
  %expr = hlfir.as_expr %c#0 : (!fir.box<!fir.array<?x!fir.char<2,?>>>) -> !hlfir.expr<?x!fir.char<2,?>>
  %c42 = arith.constant 42 : index
  %elem = hlfir.apply %expr, %c42 typeparams %l: (!hlfir.expr<?x!fir.char<2,?>>, index, index) -> !hlfir.expr<!fir.char<2,?>>
  return
}
// CHECK-LABEL:   func.func @character_apply(
// CHECK:    %[[VAL_7:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) typeparams %[[VAL_1:.*]] {uniq_name = ".tmp"}
// CHECK:    %[[VAL_11:.*]] = arith.constant 42 : index
// CHECK:    %[[VAL_12:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_11]])  typeparams %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<2,?>>>, index, index) -> !fir.boxchar<2>
// CHECK:    %[[VAL_13:.*]] = arith.constant false
// CHECK:    %[[VAL_14:.*]] = fir.undefined tuple<!fir.boxchar<2>, i1>
// CHECK:    %[[VAL_15:.*]] = fir.insert_value %[[VAL_14]], %[[VAL_13]], [1 : index] : (tuple<!fir.boxchar<2>, i1>, i1) -> tuple<!fir.boxchar<2>, i1>
// CHECK:    %[[VAL_16:.*]] = fir.insert_value %[[VAL_15]], %[[VAL_12]], [0 : index] : (tuple<!fir.boxchar<2>, i1>, !fir.boxchar<2>) -> tuple<!fir.boxchar<2>, i1>
