// RUN: fir-opt %s -opt-bufferization | FileCheck %s

func.func @_QFPtest(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> !fir.logical<4> {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %0 = fir.shape %c4, %c7 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFtestEb"} : (!fir.ref<!fir.array<4x7xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x7xi32>>, !fir.ref<!fir.array<4x7xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFFtestErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca !fir.logical<4> {bindc_name = "test", uniq_name = "_QFFtestEtest"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFFtestEtest"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %5:2 = hlfir.declare %arg2 {uniq_name = "_QFFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = fir.load %2#0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> i64
  %8 = fir.shape %c7 : (index) -> !fir.shape<1>
  %9 = hlfir.designate %1#0 (%7, %c1:%c7:%c1)  shape %8 : (!fir.ref<!fir.array<4x7xi32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<7xi32>>
  %10 = fir.load %5#0 : !fir.ref<i32>
  %11 = hlfir.elemental %8 unordered : (!fir.shape<1>) -> !hlfir.expr<7x!fir.logical<4>> {
  ^bb0(%arg3: index):
    %14 = hlfir.designate %9 (%arg3)  : (!fir.box<!fir.array<7xi32>>, index) -> !fir.ref<i32>
    %15 = fir.load %14 : !fir.ref<i32>
    %16 = arith.cmpi sge, %15, %10 : i32
    %17 = fir.convert %16 : (i1) -> !fir.logical<4>
    hlfir.yield_element %17 : !fir.logical<4>
  }
  %12 = hlfir.all %11 : (!hlfir.expr<7x!fir.logical<4>>) -> !fir.logical<4>
  hlfir.assign %12 to %4#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
  hlfir.destroy %11 : !hlfir.expr<7x!fir.logical<4>>
  %13 = fir.load %4#1 : !fir.ref<!fir.logical<4>>
  return %13 : !fir.logical<4>
}
// CHECK-LABEL:  func.func @_QFPtest(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> !fir.logical<4> {
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c4 = arith.constant 4 : index
// CHECK-NEXT:     %c7 = arith.constant 7 : index
// CHECK-NEXT:     %[[V0:.*]] = fir.shape %c4, %c7 : (index, index) -> !fir.shape<2>
// CHECK-NEXT:     %[[V1:.*]]:2 = hlfir.declare %arg0(%[[V0]])
// CHECK-NEXT:     %[[V2:.*]]:2 = hlfir.declare %arg1
// CHECK-NEXT:     %[[V3:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:     %[[V4:.*]]:2 = hlfir.declare %[[V3]]
// CHECK-NEXT:     %[[V5:.*]]:2 = hlfir.declare %arg2
// CHECK-NEXT:     %[[V6:.*]] = fir.load %[[V2]]#0 : !fir.ref<i32>
// CHECK-NEXT:     %[[V7:.*]] = fir.convert %[[V6]] : (i32) -> i64
// CHECK-NEXT:     %[[V8:.*]] = fir.shape %c7 : (index) -> !fir.shape<1>
// CHECK-NEXT:     %[[V9:.*]] = hlfir.designate %[[V1]]#0 (%[[V7]], %c1:%c7:%c1)  shape %[[V8]] : (!fir.ref<!fir.array<4x7xi32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<7xi32>>
// CHECK-NEXT:     %[[V10:.*]] = fir.load %[[V5]]#0 : !fir.ref<i32>
// CHECK-NEXT:     %[[V11:.*]] = fir.do_loop %arg3 = %c1 to %c7 step %c1 iter_args(%arg4 = %true) -> (i1) {
// CHECK-NEXT:       %[[V14:.*]] = hlfir.designate %[[V9]] (%arg3)  : (!fir.box<!fir.array<7xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:       %[[V15:.*]] = fir.load %[[V14]] : !fir.ref<i32>
// CHECK-NEXT:       %[[V16:.*]] = arith.cmpi sge, %[[V15]], %[[V10]] : i32
// CHECK-NEXT:       %[[V17:.*]] = arith.andi %arg4, %[[V16]] : i1
// CHECK-NEXT:       fir.result %[[V17]] : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V12:.*]] = fir.convert %[[V11]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:     hlfir.assign %[[V12]] to %[[V4]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK-NEXT:     %[[V13:.*]] = fir.load %[[V4]]#1 : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:     return %[[V13]] : !fir.logical<4>


func.func @_QFPtest_dim(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> !fir.array<4x!fir.logical<4>> {
  %c2_i32 = arith.constant 2 : i32
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %0 = fir.shape %c4, %c7 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFtestEb"} : (!fir.ref<!fir.array<4x7xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x7xi32>>, !fir.ref<!fir.array<4x7xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFFtestErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca !fir.array<4x!fir.logical<4>> {bindc_name = "test", uniq_name = "_QFFtestEtest"}
  %4 = fir.shape %c4 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %3(%4) {uniq_name = "_QFFtestEtest"} : (!fir.ref<!fir.array<4x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4x!fir.logical<4>>>, !fir.ref<!fir.array<4x!fir.logical<4>>>)
  %6:2 = hlfir.declare %arg2 {uniq_name = "_QFFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = hlfir.designate %1#0 (%c1:%c4:%c1, %c1:%c7:%c1)  shape %0 : (!fir.ref<!fir.array<4x7xi32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<4x7xi32>>
  %8 = fir.load %6#0 : !fir.ref<i32>
  %9 = hlfir.elemental %0 unordered : (!fir.shape<2>) -> !hlfir.expr<4x7x!fir.logical<4>> {
  ^bb0(%arg3: index, %arg4: index):
    %12 = hlfir.designate %7 (%arg3, %arg4)  : (!fir.ref<!fir.array<4x7xi32>>, index, index) -> !fir.ref<i32>
    %13 = fir.load %12 : !fir.ref<i32>
    %14 = arith.cmpi sge, %13, %8 : i32
    %15 = fir.convert %14 : (i1) -> !fir.logical<4>
    hlfir.yield_element %15 : !fir.logical<4>
  }
  %10 = hlfir.all %9 dim %c2_i32 : (!hlfir.expr<4x7x!fir.logical<4>>, i32) -> !hlfir.expr<4x!fir.logical<4>>
  hlfir.assign %10 to %5#0 : !hlfir.expr<4x!fir.logical<4>>, !fir.ref<!fir.array<4x!fir.logical<4>>>
  hlfir.destroy %10 : !hlfir.expr<4x!fir.logical<4>>
  hlfir.destroy %9 : !hlfir.expr<4x7x!fir.logical<4>>
  %11 = fir.load %5#1 : !fir.ref<!fir.array<4x!fir.logical<4>>>
  return %11 : !fir.array<4x!fir.logical<4>>
}
// CHECK-LABEL:  func.func @_QFPtest_dim(
// CHECK: %10 = hlfir.all %9 dim %c2_i32