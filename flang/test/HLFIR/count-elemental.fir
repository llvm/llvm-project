// RUN: fir-opt %s -opt-bufferization | FileCheck %s

func.func @_QFPtest(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> i32 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %0 = fir.shape %c4, %c7 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFtestEb"} : (!fir.ref<!fir.array<4x7xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x7xi32>>, !fir.ref<!fir.array<4x7xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFFtestErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca i32 {bindc_name = "test", uniq_name = "_QFFtestEtest"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFFtestEtest"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
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
  %12 = hlfir.count %11 : (!hlfir.expr<7x!fir.logical<4>>) -> i32
  hlfir.assign %12 to %4#0 : i32, !fir.ref<i32>
  hlfir.destroy %11 : !hlfir.expr<7x!fir.logical<4>>
  %13 = fir.load %4#1 : !fir.ref<i32>
  return %13 : i32
}
// CHECK-LABEL:  func.func @_QFPtest(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> i32 {
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c4 = arith.constant 4 : index
// CHECK-NEXT:     %c7 = arith.constant 7 : index
// CHECK-NEXT:     %[[V0:.*]] = fir.shape %c4, %c7 : (index, index) -> !fir.shape<2>
// CHECK-NEXT:     %[[V1:.*]]:2 = hlfir.declare %arg0(%[[V0]])
// CHECK-NEXT:     %[[V2:.*]]:2 = hlfir.declare %arg1
// CHECK-NEXT:     %[[V3:.*]] = fir.alloca i32
// CHECK-NEXT:     %[[V4:.*]]:2 = hlfir.declare %[[V3]]
// CHECK-NEXT:     %[[V5:.*]]:2 = hlfir.declare %arg2
// CHECK-NEXT:     %[[V6:.*]] = fir.load %[[V2]]#0 : !fir.ref<i32>
// CHECK-NEXT:     %[[V7:.*]] = fir.convert %[[V6]] : (i32) -> i64
// CHECK-NEXT:     %[[V8:.*]] = fir.shape %c7 : (index) -> !fir.shape<1>
// CHECK-NEXT:     %[[V9:.*]] = hlfir.designate %[[V1]]#0 (%[[V7]], %c1:%c7:%c1)  shape %[[V8]] : (!fir.ref<!fir.array<4x7xi32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<7xi32>>
// CHECK-NEXT:     %[[V10:.*]] = fir.load %[[V5]]#0 : !fir.ref<i32>
// CHECK-NEXT:     %[[V11:.*]] = fir.do_loop %arg3 = %c1 to %c7 step %c1 iter_args(%arg4 = %c0_i32) -> (i32) {
// CHECK-NEXT:       %[[V13:.*]] = hlfir.designate %[[V9]] (%arg3)  : (!fir.box<!fir.array<7xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:       %[[V14:.*]] = fir.load %[[V13]] : !fir.ref<i32>
// CHECK-NEXT:       %[[V15:.*]] = arith.cmpi sge, %[[V14]], %[[V10]] : i32
// CHECK-NEXT:       %[[V16:.*]] = arith.addi %arg4, %c1_i32 : i32
// CHECK-NEXT:       %[[V17:.*]] = arith.select %[[V15]], %[[V16]], %arg4 : i32
// CHECK-NEXT:       fir.result %[[V17]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     hlfir.assign %[[V11]] to %[[V4]]#0 : i32, !fir.ref<i32>
// CHECK-NEXT:     %[[V12:.*]] = fir.load %[[V4]]#1 : !fir.ref<i32>
// CHECK-NEXT:     return %[[V12]] : i32

func.func @_QFPtest_kind2(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> i16 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %0 = fir.shape %c4, %c7 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFtestEb"} : (!fir.ref<!fir.array<4x7xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x7xi32>>, !fir.ref<!fir.array<4x7xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFFtestErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca i16 {bindc_name = "test", uniq_name = "_QFFtestEtest"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFFtestEtest"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
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
  %12 = hlfir.count %11 : (!hlfir.expr<7x!fir.logical<4>>) -> i16
  hlfir.assign %12 to %4#0 : i16, !fir.ref<i16>
  hlfir.destroy %11 : !hlfir.expr<7x!fir.logical<4>>
  %13 = fir.load %4#1 : !fir.ref<i16>
  return %13 : i16
}
// CHECK-LABEL:  func.func @_QFPtest_kind2(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> i16 {
// CHECK-NEXT:     %c1_i16 = arith.constant 1 : i16
// CHECK-NEXT:     %c0_i16 = arith.constant 0 : i16
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c4 = arith.constant 4 : index
// CHECK-NEXT:     %c7 = arith.constant 7 : index
// CHECK-NEXT:     %[[V0:.*]] = fir.shape %c4, %c7 : (index, index) -> !fir.shape<2>
// CHECK-NEXT:     %[[V1:.*]]:2 = hlfir.declare %arg0(%[[V0]])
// CHECK-NEXT:     %[[V2:.*]]:2 = hlfir.declare %arg1
// CHECK-NEXT:     %[[V3:.*]] = fir.alloca i16
// CHECK-NEXT:     %[[V4:.*]]:2 = hlfir.declare %[[V3]]
// CHECK-NEXT:     %[[V5:.*]]:2 = hlfir.declare %arg2
// CHECK-NEXT:     %[[V6:.*]] = fir.load %[[V2]]#0 : !fir.ref<i32>
// CHECK-NEXT:     %[[V7:.*]] = fir.convert %[[V6]] : (i32) -> i64
// CHECK-NEXT:     %[[V8:.*]] = fir.shape %c7 : (index) -> !fir.shape<1>
// CHECK-NEXT:     %[[V9:.*]] = hlfir.designate %[[V1]]#0 (%[[V7]], %c1:%c7:%c1)  shape %[[V8]] : (!fir.ref<!fir.array<4x7xi32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<7xi32>>
// CHECK-NEXT:     %[[V10:.*]] = fir.load %[[V5]]#0 : !fir.ref<i32>
// CHECK-NEXT:     %[[V11:.*]] = fir.do_loop %arg3 = %c1 to %c7 step %c1 iter_args(%arg4 = %c0_i16) -> (i16) {
// CHECK-NEXT:       %[[V13:.*]] = hlfir.designate %[[V9]] (%arg3)  : (!fir.box<!fir.array<7xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:       %[[V14:.*]] = fir.load %[[V13]] : !fir.ref<i32>
// CHECK-NEXT:       %[[V15:.*]] = arith.cmpi sge, %[[V14]], %[[V10]] : i32
// CHECK-NEXT:       %[[V16:.*]] = arith.addi %arg4, %c1_i16 : i16
// CHECK-NEXT:       %[[V17:.*]] = arith.select %[[V15]], %[[V16]], %arg4 : i16
// CHECK-NEXT:       fir.result %[[V17]] : i16
// CHECK-NEXT:     }
// CHECK-NEXT:     hlfir.assign %[[V11]] to %[[V4]]#0 : i16, !fir.ref<i16>
// CHECK-NEXT:     %[[V12:.*]] = fir.load %[[V4]]#1 : !fir.ref<i16>
// CHECK-NEXT:     return %[[V12]] : i16

func.func @_QFPtest_dim(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> !fir.array<7xi32> {
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %0 = fir.shape %c4, %c7 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFtestEb"} : (!fir.ref<!fir.array<4x7xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x7xi32>>, !fir.ref<!fir.array<4x7xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFFtestErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca !fir.array<7xi32> {bindc_name = "test", uniq_name = "_QFFtestEtest"}
  %4 = fir.shape %c7 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %3(%4) {uniq_name = "_QFFtestEtest"} : (!fir.ref<!fir.array<7xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<7xi32>>, !fir.ref<!fir.array<7xi32>>)
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
  %10 = hlfir.count %9 dim %c1_i32 : (!hlfir.expr<4x7x!fir.logical<4>>, i32) -> !hlfir.expr<7xi32>
  hlfir.assign %10 to %5#0 : !hlfir.expr<7xi32>, !fir.ref<!fir.array<7xi32>>
  hlfir.destroy %10 : !hlfir.expr<7xi32>
  hlfir.destroy %9 : !hlfir.expr<4x7x!fir.logical<4>>
  %11 = fir.load %5#1 : !fir.ref<!fir.array<7xi32>>
  return %11 : !fir.array<7xi32>
}
// CHECK-LABEL:  func.func @_QFPtest_dim(
// CHECK: %{{.*}} = hlfir.count %{{.*}} dim %c1_i32


func.func @_QFPtest_multi(%arg0: !fir.ref<!fir.array<4x7x2xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> i32 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %c2 = arith.constant 2 : index
  %0 = fir.shape %c4, %c7, %c2 : (index, index, index) -> !fir.shape<3>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFtestEb"} : (!fir.ref<!fir.array<4x7x2xi32>>, !fir.shape<3>) -> (!fir.ref<!fir.array<4x7x2xi32>>, !fir.ref<!fir.array<4x7x2xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFFtestErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca i32 {bindc_name = "test", uniq_name = "_QFFtestEtest"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFFtestEtest"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg2 {uniq_name = "_QFFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = hlfir.designate %1#0 (%c1:%c4:%c1, %c1:%c7:%c1, %c1:%c2:%c1)  shape %0 : (!fir.ref<!fir.array<4x7x2xi32>>, index, index, index, index, index, index, index, index, index, !fir.shape<3>) -> !fir.ref<!fir.array<4x7x2xi32>>
  %7 = fir.load %5#0 : !fir.ref<i32>
  %8 = hlfir.elemental %0 unordered : (!fir.shape<3>) -> !hlfir.expr<4x7x2x!fir.logical<4>> {
  ^bb0(%arg3: index, %arg4: index, %arg5: index):
    %11 = hlfir.designate %6 (%arg3, %arg4, %arg5)  : (!fir.ref<!fir.array<4x7x2xi32>>, index, index, index) -> !fir.ref<i32>
    %12 = fir.load %11 : !fir.ref<i32>
    %13 = arith.cmpi sge, %12, %7 : i32
    %14 = fir.convert %13 : (i1) -> !fir.logical<4>
    hlfir.yield_element %14 : !fir.logical<4>
  }
  %9 = hlfir.count %8 : (!hlfir.expr<4x7x2x!fir.logical<4>>) -> i32
  hlfir.assign %9 to %4#0 : i32, !fir.ref<i32>
  hlfir.destroy %8 : !hlfir.expr<4x7x2x!fir.logical<4>>
  %10 = fir.load %4#1 : !fir.ref<i32>
  return %10 : i32
}
// CHECK-LABEL:  func.func @_QFPtest_multi(%arg0: !fir.ref<!fir.array<4x7x2xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> i32 {
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c4 = arith.constant 4 : index
// CHECK-NEXT:     %c7 = arith.constant 7 : index
// CHECK-NEXT:     %c2 = arith.constant 2 : index
// CHECK-NEXT:     %[[V0:.*]] = fir.shape %c4, %c7, %c2 : (index, index, index) -> !fir.shape<3>
// CHECK-NEXT:     %[[V1:.*]]:2 = hlfir.declare %arg0(%[[V0]]) {uniq_name = "_QFFtestEb"} : (!fir.ref<!fir.array<4x7x2xi32>>, !fir.shape<3>) -> (!fir.ref<!fir.array<4x7x2xi32>>, !fir.ref<!fir.array<4x7x2xi32>>)
// CHECK-NEXT:     %[[V2:.*]]:2 = hlfir.declare %arg1 {uniq_name = "_QFFtestErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK-NEXT:     %[[V3:.*]] = fir.alloca i32 {bindc_name = "test", uniq_name = "_QFFtestEtest"}
// CHECK-NEXT:     %[[V4:.*]]:2 = hlfir.declare %[[V3]] {uniq_name = "_QFFtestEtest"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK-NEXT:     %[[V5:.*]]:2 = hlfir.declare %arg2 {uniq_name = "_QFFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK-NEXT:     %[[V6:.*]] = hlfir.designate %[[V1]]#0 (%c1:%c4:%c1, %c1:%c7:%c1, %c1:%c2:%c1)  shape %[[V0]] : (!fir.ref<!fir.array<4x7x2xi32>>, index, index, index, index, index, index, index, index, index, !fir.shape<3>) -> !fir.ref<!fir.array<4x7x2xi32>>
// CHECK-NEXT:     %[[V7:.*]] = fir.load %[[V5]]#0 : !fir.ref<i32>
// CHECK-NEXT:     %[[V8:.*]] = fir.do_loop %arg3 = %c1 to %c2 step %c1 iter_args(%arg4 = %c0_i32) -> (i32) {
// CHECK-NEXT:       %[[V10:.*]] = fir.do_loop %arg5 = %c1 to %c7 step %c1 iter_args(%arg6 = %arg4) -> (i32) {
// CHECK-NEXT:         %[[V11:.*]] = fir.do_loop %arg7 = %c1 to %c4 step %c1 iter_args(%arg8 = %arg6) -> (i32) {
// CHECK-NEXT:           %[[V12:.*]] = hlfir.designate %[[V6]] (%arg7, %arg5, %arg3)  : (!fir.ref<!fir.array<4x7x2xi32>>, index, index, index) -> !fir.ref<i32>
// CHECK-NEXT:           %[[V13:.*]] = fir.load %[[V12]] : !fir.ref<i32>
// CHECK-NEXT:           %[[V14:.*]] = arith.cmpi sge, %[[V13]], %[[V7]] : i32
// CHECK-NEXT:           %[[V15:.*]] = arith.addi %arg8, %c1_i32 : i32
// CHECK-NEXT:           %[[V16:.*]] = arith.select %[[V14]], %[[V15]], %arg8 : i32
// CHECK-NEXT:           fir.result %[[V16]] : i32
// CHECK-NEXT:         }
// CHECK-NEXT:         fir.result %[[V11]] : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       fir.result %[[V10]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     hlfir.assign %[[V8]] to %[[V4]]#0 : i32, !fir.ref<i32>
// CHECK-NEXT:     %[[V9:.*]] = fir.load %[[V4]]#1 : !fir.ref<i32>
// CHECK-NEXT:     return %[[V9]] : i32





func.func @_QFPtest_rec_sum(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> i32 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %0 = fir.shape %c4, %c7 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFtestEb"} : (!fir.ref<!fir.array<4x7xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x7xi32>>, !fir.ref<!fir.array<4x7xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFFtestErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca i32 {bindc_name = "test", uniq_name = "_QFFtestEtest"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFFtestEtest"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg2 {uniq_name = "_QFFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = fir.load %2#0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> i64
  %8 = fir.shape %c7 : (index) -> !fir.shape<1>
  %9 = hlfir.designate %1#0 (%7, %c1:%c7:%c1)  shape %8 : (!fir.ref<!fir.array<4x7xi32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<7xi32>>
  %10 = fir.load %5#0 : !fir.ref<i32>
  %11 = hlfir.elemental %8 unordered : (!fir.shape<1>) -> !hlfir.expr<7xi32> {
  ^bb0(%arg3: index):
    %15 = hlfir.designate %9 (%arg3)  : (!fir.box<!fir.array<7xi32>>, index) -> !fir.ref<i32>
    %16 = fir.load %15 : !fir.ref<i32>
    hlfir.yield_element %16 : i32
  }
  %12 = hlfir.elemental %8 unordered : (!fir.shape<1>) -> !hlfir.expr<7x!fir.logical<4>> {
  ^bb0(%arg3: index):
    %15 = hlfir.sum %11 : (!hlfir.expr<7xi32>) -> i32
    %16 = arith.cmpi sge, %15, %10 : i32
    %17 = fir.convert %16 : (i1) -> !fir.logical<4>
    hlfir.yield_element %17 : !fir.logical<4>
  }
  %13 = hlfir.count %12 : (!hlfir.expr<7x!fir.logical<4>>) -> i32
  hlfir.assign %13 to %4#0 : i32, !fir.ref<i32>
  hlfir.destroy %12 : !hlfir.expr<7x!fir.logical<4>>
  hlfir.destroy %11 : !hlfir.expr<7xi32>
  %14 = fir.load %4#1 : !fir.ref<i32>
  return %14 : i32
}
// CHECK-LABEL:  func.func @_QFPtest_rec_sum(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> i32 {
// CHECK:    %[[V12:.*]] = fir.do_loop %arg3 = %c1 to %c7 step %c1 iter_args(%arg4 = %c0_i32) -> (i32) {
// CHECK:      %[[V14:.*]] = hlfir.sum %[[V11]] : (!hlfir.expr<7xi32>) -> i32
// CHECK:      %[[V15:.*]] = arith.cmpi sge, %[[V14]], %[[V10]] : i32
// CHECK:      %[[V16:.*]] = arith.addi %arg4, %c1_i32 : i32
// CHECK:      %[[V17:.*]] = arith.select %[[V15]], %[[V16]], %arg4 : i32
// CHECK:      fir.result %[[V17]] : i32
// CHECK:    }




func.func @_QFPtest_rec_count(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> i32 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %0 = fir.shape %c4, %c7 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFtestEb"} : (!fir.ref<!fir.array<4x7xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x7xi32>>, !fir.ref<!fir.array<4x7xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFFtestErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca i32 {bindc_name = "test", uniq_name = "_QFFtestEtest"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFFtestEtest"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5:2 = hlfir.declare %arg2 {uniq_name = "_QFFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = fir.load %2#0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> i64
  %8 = fir.shape %c7 : (index) -> !fir.shape<1>
  %9 = hlfir.designate %1#0 (%7, %c1:%c7:%c1)  shape %8 : (!fir.ref<!fir.array<4x7xi32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<7xi32>>
  %10 = fir.load %5#0 : !fir.ref<i32>
  %11 = hlfir.elemental %8 unordered : (!fir.shape<1>) -> !hlfir.expr<7x!fir.logical<4>> {
  ^bb0(%arg3: index):
    %15 = hlfir.designate %9 (%arg3)  : (!fir.box<!fir.array<7xi32>>, index) -> !fir.ref<i32>
    %16 = fir.load %15 : !fir.ref<i32>
    %17 = arith.cmpi sge, %16, %10 : i32
    %18 = fir.convert %17 : (i1) -> !fir.logical<4>
    hlfir.yield_element %18 : !fir.logical<4>
  }
  %12 = hlfir.elemental %8 unordered : (!fir.shape<1>) -> !hlfir.expr<7x!fir.logical<4>> {
  ^bb0(%arg3: index):
    %15 = hlfir.count %11 : (!hlfir.expr<7x!fir.logical<4>>) -> i32
    %16 = arith.cmpi sge, %15, %10 : i32
    %17 = fir.convert %16 : (i1) -> !fir.logical<4>
    hlfir.yield_element %17 : !fir.logical<4>
  }
  %13 = hlfir.count %12 : (!hlfir.expr<7x!fir.logical<4>>) -> i32
  hlfir.assign %13 to %4#0 : i32, !fir.ref<i32>
  hlfir.destroy %12 : !hlfir.expr<7x!fir.logical<4>>
  hlfir.destroy %11 : !hlfir.expr<7x!fir.logical<4>>
  %14 = fir.load %4#1 : !fir.ref<i32>
  return %14 : i32
}
// CHECK-LABEL:  func.func @_QFPtest_rec_count(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> i32 {
// CHECK:    %[[V11:.*]] = fir.do_loop %arg3 = %c1 to %c7 step %c1 iter_args(%arg4 = %c0_i32) -> (i32) {
// CHECK:      %[[V13:.*]] = fir.do_loop %arg5 = %c1 to %c7 step %c1 iter_args(%arg6 = %c0_i32) -> (i32) {
// CHECK:        %[[V17:.*]] = hlfir.designate %[[V9]] (%arg5)  : (!fir.box<!fir.array<7xi32>>, index) -> !fir.ref<i32>
// CHECK:        %[[V18:.*]] = fir.load %[[V17]] : !fir.ref<i32>
// CHECK:        %[[V19:.*]] = arith.cmpi sge, %[[V18]], %[[V10]] : i32
// CHECK:        %[[V20:.*]] = arith.addi %arg6, %c1_i32 : i32
// CHECK:        %[[V21:.*]] = arith.select %[[V19]], %[[V20]], %arg6 : i32
// CHECK:        fir.result %[[V21]] : i32
// CHECK:      }
// CHECK:      %[[V14:.*]] = arith.cmpi sge, %[[V13]], %[[V10]] : i32
// CHECK:      %[[V15:.*]] = arith.addi %arg4, %c1_i32 : i32
// CHECK:      %[[V16:.*]] = arith.select %[[V14]], %[[V15]], %arg4 : i32
// CHECK:      fir.result %[[V16]] : i32
// CHECK:    }
