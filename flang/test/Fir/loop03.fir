// Test the reduction semantics of fir.do_loop
// RUN: fir-opt %s | FileCheck %s

func.func @reduction() {
  %bound = arith.constant 10 : index
  %step = arith.constant 1 : index
  %sum = fir.alloca i32
  %red = fir.reduce %sum {name = "sum"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK: %[[VAL_0:.*]] = fir.alloca i32
// CHECK: %[[VAL_1:.*]] = fir.reduce %[[VAL_0]] {name = "sum"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK: fir.do_loop %[[VAL_2:.*]] = %[[VAL_3:.*]] to %[[VAL_4:.*]] step %[[VAL_5:.*]] unordered reduce(#fir.reduce_attr<add> -> %[[VAL_1]] : !fir.ref<i32>) {
  fir.do_loop %iv = %step to %bound step %step unordered reduce(#fir.reduce_attr<add> -> %red : !fir.ref<i32>) {
    %index = fir.convert %iv : (index) -> i32
    %1 = fir.load %sum : !fir.ref<i32>
    %2 = arith.addi %index, %1 : i32
    fir.store %2 to %sum : !fir.ref<i32>
  }
  return
}
