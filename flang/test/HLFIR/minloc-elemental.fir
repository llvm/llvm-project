// RUN: fir-opt %s -opt-bufferization | FileCheck %s

func.func @_QPtest(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"}, %arg1: !fir.ref<i32> {fir.bindc_name = "val"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "m"}) {
  %c0 = arith.constant 0 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtestEarray"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFtestEm"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.load %2#0 : !fir.ref<i32>
  %4:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %5 = fir.shape %4#1 : (index) -> !fir.shape<1>
  %6 = hlfir.elemental %5 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%arg3: index):
    %8 = hlfir.designate %0#0 (%arg3)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %9 = fir.load %8 : !fir.ref<i32>
    %10 = arith.cmpi sge, %9, %3 : i32
    %11 = fir.convert %10 : (i1) -> !fir.logical<4>
    hlfir.yield_element %11 : !fir.logical<4>
  }
  %7 = hlfir.minloc %0#0 mask %6 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  hlfir.assign %7 to %1#0 : !hlfir.expr<1xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %7 : !hlfir.expr<1xi32>
  hlfir.destroy %6 : !hlfir.expr<?x!fir.logical<4>>
  return
}
// CHECK-LABEL: func.func @_QPtest(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"}, %arg1: !fir.ref<i32> {fir.bindc_name = "val"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "m"}) {
// CHECK-NEXT:    %c2147483647_i32 = arith.constant 2147483647 : i32
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %[[V0:.*]] = fir.alloca i32
// CHECK-NEXT:    %[[RES:.*]] = fir.alloca !fir.array<1xi32>
// CHECK-NEXT:    %[[V1:.*]]:2 = hlfir.declare %arg0 {uniq_name = "_QFtestEarray"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %[[V2:.*]]:2 = hlfir.declare %arg2 {uniq_name = "_QFtestEm"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %[[V3:.*]]:2 = hlfir.declare %arg1 {uniq_name = "_QFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK-NEXT:    %[[V4:.*]] = fir.load %[[V3]]#0 : !fir.ref<i32>
// CHECK-NEXT:    %[[V8:.*]] = hlfir.designate %[[RES]] (%c1) : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:    fir.store %c0_i32 to %[[V8]] : !fir.ref<i32>
// CHECK-NEXT:    fir.store %c0_i32 to %[[V0]] : !fir.ref<i32>
// CHECK-NEXT:    %[[V9:.*]]:3 = fir.box_dims %[[V1]]#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK-NEXT:    %[[V10:.*]] = arith.subi %[[V9]]#1, %c1 : index
// CHECK-NEXT:    %[[V11:.*]] = fir.do_loop %arg3 = %c0 to %[[V10]] step %c1 iter_args(%arg4 = %c2147483647_i32) -> (i32) {
// CHECK-NEXT:      %[[V14:.*]] = arith.addi %arg3, %c1 : index
// CHECK-NEXT:      %[[V15:.*]] = hlfir.designate %[[V1]]#0 (%[[V14]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:      %[[V16:.*]] = fir.load %[[V15]] : !fir.ref<i32>
// CHECK-NEXT:      %[[V17:.*]] = arith.cmpi sge, %[[V16]], %[[V4]] : i32
// CHECK-NEXT:      %[[V18:.*]] = fir.if %[[V17]] -> (i32) {
// CHECK-NEXT:        fir.store %c1_i32 to %[[V0]] : !fir.ref<i32>
// CHECK-NEXT:        %[[DIMS:.*]]:3 = fir.box_dims %[[V1]]#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK-NEXT:        %[[SUB:.*]] = arith.subi %[[DIMS]]#0, %c1 : index
// CHECK-NEXT:        %[[ADD:.*]] = arith.addi %[[V14]], %[[SUB]] : index
// CHECK-NEXT:        %[[V19:.*]] = hlfir.designate %[[V1]]#0 (%[[ADD]]) : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:        %[[V20:.*]] = fir.load %[[V19]] : !fir.ref<i32>
// CHECK-NEXT:        %[[V21:.*]] = arith.cmpi slt, %[[V20]], %arg4 : i32
// CHECK-NEXT:        %[[V22:.*]] = fir.if %[[V21]] -> (i32) {
// CHECK-NEXT:          %[[V23:.*]] = hlfir.designate %[[RES]] (%c1) : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:          %[[V24:.*]] = fir.convert %[[V14]] : (index) -> i32
// CHECK-NEXT:          fir.store %[[V24]] to %[[V23]] : !fir.ref<i32>
// CHECK-NEXT:          fir.result %[[V20]] : i32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          fir.result %arg4 : i32
// CHECK-NEXT:        }
// CHECK-NEXT:        fir.result %[[V22]] : i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        fir.result %arg4 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      fir.result %[[V18]] : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[V12:.*]] = fir.load %[[V0]] : !fir.ref<i32>
// CHECK-NEXT:    %[[V13:.*]] = arith.cmpi eq, %[[V12]], %c1_i32 : i32
// CHECK-NEXT:    fir.if %[[V13]] {
// CHECK-NEXT:      %[[V14:.*]] = arith.cmpi eq, %[[V11]], %c2147483647_i32 : i32
// CHECK-NEXT:      fir.if %[[V14]] {
// CHECK-NEXT:        %[[V15:.*]] = hlfir.designate %[[RES]] (%c1) : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:        fir.store %c1_i32 to %[[V15]] : !fir.ref<i32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[BD:.*]]:3 = fir.box_dims %[[V2]]#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK-NEXT:    fir.do_loop %arg3 = %c1 to %[[BD]]#1 step %c1 unordered {
// CHECK-NEXT:      %[[V13:.*]] = hlfir.designate %[[RES]] (%arg3)  : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:      %[[V14:.*]] = fir.load %[[V13]] : !fir.ref<i32>
// CHECK-NEXT:      %[[V15:.*]] = hlfir.designate %[[V2]]#0 (%arg3)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:      hlfir.assign %[[V14]] to %[[V15]] : i32, !fir.ref<i32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }


func.func @_QPtest_kind2(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"}, %arg1: !fir.ref<i32> {fir.bindc_name = "val"}, %arg2: !fir.box<!fir.array<?xi16>> {fir.bindc_name = "m"}) {
  %c0 = arith.constant 0 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtestEarray"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFtestEm"} : (!fir.box<!fir.array<?xi16>>) -> (!fir.box<!fir.array<?xi16>>, !fir.box<!fir.array<?xi16>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.load %2#0 : !fir.ref<i32>
  %4:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %5 = fir.shape %4#1 : (index) -> !fir.shape<1>
  %6 = hlfir.elemental %5 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%arg3: index):
    %8 = hlfir.designate %0#0 (%arg3)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %9 = fir.load %8 : !fir.ref<i32>
    %10 = arith.cmpi sge, %9, %3 : i32
    %11 = fir.convert %10 : (i1) -> !fir.logical<4>
    hlfir.yield_element %11 : !fir.logical<4>
  }
  %7 = hlfir.minloc %0#0 mask %6 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi16>
  hlfir.assign %7 to %1#0 : !hlfir.expr<1xi16>, !fir.box<!fir.array<?xi16>>
  hlfir.destroy %7 : !hlfir.expr<1xi16>
  hlfir.destroy %6 : !hlfir.expr<?x!fir.logical<4>>
  return
}
// CHECK-LABEL:  func.func @_QPtest_kind2(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"}, %arg1: !fir.ref<i32> {fir.bindc_name = "val"}, %arg2: !fir.box<!fir.array<?xi16>> {fir.bindc_name = "m"}) {
// CHECK-NEXT:    %c2147483647_i32 = arith.constant 2147483647 : i32
// CHECK-NEXT:    %c1_i16 = arith.constant 1 : i16
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i16 = arith.constant 0 : i16
// CHECK-NEXT:    %[[V0:.*]] = fir.alloca i16
// CHECK-NEXT:    %[[RES:.*]] = fir.alloca !fir.array<1xi16>
// CHECK-NEXT:    %[[V1:.*]]:2 = hlfir.declare %arg0 {uniq_name = "_QFtestEarray"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %[[V2:.*]]:2 = hlfir.declare %arg2 {uniq_name = "_QFtestEm"} : (!fir.box<!fir.array<?xi16>>) -> (!fir.box<!fir.array<?xi16>>, !fir.box<!fir.array<?xi16>>)
// CHECK-NEXT:    %[[V3:.*]]:2 = hlfir.declare %arg1 {uniq_name = "_QFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK-NEXT:    %[[V4:.*]] = fir.load %[[V3]]#0 : !fir.ref<i32>
// CHECK-NEXT:    %[[V8:.*]] = hlfir.designate %[[RES]] (%c1) : (!fir.ref<!fir.array<1xi16>>, index) -> !fir.ref<i16>
// CHECK-NEXT:    fir.store %c0_i16 to %[[V8]] : !fir.ref<i16>
// CHECK-NEXT:    fir.store %c0_i16 to %[[V0]] : !fir.ref<i16>
// CHECK-NEXT:    %[[V9:.*]]:3 = fir.box_dims %[[V1]]#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK-NEXT:    %[[V10:.*]] = arith.subi %[[V9]]#1, %c1 : index
// CHECK-NEXT:    %[[V11:.*]] = fir.do_loop %arg3 = %c0 to %[[V10]] step %c1 iter_args(%arg4 = %c2147483647_i32) -> (i32) {
// CHECK-NEXT:      %[[V14:.*]] = arith.addi %arg3, %c1 : index
// CHECK-NEXT:      %[[V15:.*]] = hlfir.designate %[[V1]]#0 (%[[V14]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:      %[[V16:.*]] = fir.load %[[V15]] : !fir.ref<i32>
// CHECK-NEXT:      %[[V17:.*]] = arith.cmpi sge, %[[V16]], %[[V4]] : i32
// CHECK-NEXT:      %[[V18:.*]] = fir.if %[[V17]] -> (i32) {
// CHECK-NEXT:        fir.store %c1_i16 to %[[V0]] : !fir.ref<i16>
// CHECK-NEXT:        %[[DIMS:.*]]:3 = fir.box_dims %[[V1]]#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK-NEXT:        %[[SUB:.*]] = arith.subi %[[DIMS]]#0, %c1 : index
// CHECK-NEXT:        %[[ADD:.*]] = arith.addi %[[V14]], %[[SUB]] : index
// CHECK-NEXT:        %[[V19:.*]] = hlfir.designate %[[V1]]#0 (%[[ADD]]) : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:        %[[V20:.*]] = fir.load %[[V19]] : !fir.ref<i32>
// CHECK-NEXT:        %[[V21:.*]] = arith.cmpi slt, %[[V20]], %arg4 : i32
// CHECK-NEXT:        %[[V22:.*]] = fir.if %[[V21]] -> (i32) {
// CHECK-NEXT:          %[[V23:.*]] = hlfir.designate %[[RES]] (%c1) : (!fir.ref<!fir.array<1xi16>>, index) -> !fir.ref<i16>
// CHECK-NEXT:          %[[V24:.*]] = fir.convert %[[V14]] : (index) -> i16
// CHECK-NEXT:          fir.store %[[V24]] to %[[V23]] : !fir.ref<i16>
// CHECK-NEXT:          fir.result %[[V20]] : i32
// CHECK-NEXT:        } else {
// CHECK-NEXT:          fir.result %arg4 : i32
// CHECK-NEXT:        }
// CHECK-NEXT:        fir.result %[[V22]] : i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        fir.result %arg4 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      fir.result %[[V18]] : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[V12:.*]] = fir.load %[[V0]] : !fir.ref<i16>
// CHECK-NEXT:    %[[V13:.*]] = arith.cmpi eq, %[[V12]], %c1_i16 : i16
// CHECK-NEXT:    fir.if %[[V13]] {
// CHECK-NEXT:      %[[V14:.*]] = arith.cmpi eq, %[[V11]], %c2147483647_i32 : i32
// CHECK-NEXT:      fir.if %[[V14]] {
// CHECK-NEXT:        %[[V15:.*]] = hlfir.designate %[[RES]] (%c1) : (!fir.ref<!fir.array<1xi16>>, index) -> !fir.ref<i16>
// CHECK-NEXT:        fir.store %c1_i16 to %[[V15]] : !fir.ref<i16>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[BD:.*]]:3 = fir.box_dims %[[V2]]#0, %c0 : (!fir.box<!fir.array<?xi16>>, index) -> (index, index, index)
// CHECK-NEXT:    fir.do_loop %arg3 = %c1 to %[[BD]]#1 step %c1 unordered {
// CHECK-NEXT:      %[[V13:.*]] = hlfir.designate %[[RES]] (%arg3)  : (!fir.ref<!fir.array<1xi16>>, index) -> !fir.ref<i16>
// CHECK-NEXT:      %[[V14:.*]] = fir.load %[[V13]] : !fir.ref<i16>
// CHECK-NEXT:      %[[V15:.*]] = hlfir.designate %[[V2]]#0 (%arg3)  : (!fir.box<!fir.array<?xi16>>, index) -> !fir.ref<i16>
// CHECK-NEXT:      hlfir.assign %[[V14]] to %[[V15]] : i16, !fir.ref<i16>
// CHECK-NEXT:    }
// CHECK-NEXT:    return


func.func @_QPtest_kind2_convert(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"}, %arg1: !fir.ref<i32> {fir.bindc_name = "val"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "m"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtestEarray"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFtestEm"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.load %2#0 : !fir.ref<i32>
  %4:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %5 = fir.shape %4#1 : (index) -> !fir.shape<1>
  %6 = hlfir.elemental %5 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%arg3: index):
    %10 = hlfir.designate %0#0 (%arg3)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %11 = fir.load %10 : !fir.ref<i32>
    %12 = arith.cmpi sge, %11, %3 : i32
    %13 = fir.convert %12 : (i1) -> !fir.logical<4>
    hlfir.yield_element %13 : !fir.logical<4>
  }
  %7 = hlfir.minloc %0#0 mask %6 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi16>
  %8 = fir.shape %c1 : (index) -> !fir.shape<1>
  %9 = hlfir.elemental %8 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg3: index):
    %10 = hlfir.apply %7, %arg3 : (!hlfir.expr<1xi16>, index) -> i16
    %11 = fir.convert %10 : (i16) -> i32
    hlfir.yield_element %11 : i32
  }
  hlfir.assign %9 to %1#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %9 : !hlfir.expr<?xi32>
  hlfir.destroy %7 : !hlfir.expr<1xi16>
  hlfir.destroy %6 : !hlfir.expr<?x!fir.logical<4>>
  return
}
// The minloc has other uses, not an assign that gets optimized out.
// CHECK-LABEL: _QPtest_kind2_convert
// CHECK-SAME:     (%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"}, %arg1: !fir.ref<i32> {fir.bindc_name = "val"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "m"}) {
// CHECK-NEXT:   %false = arith.constant false
// CHECK-NEXT:   %c2147483647_i32 = arith.constant 2147483647 : i32
// CHECK-NEXT:   %c1_i16 = arith.constant 1 : i16
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c0_i16 = arith.constant 0 : i16
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %[[V0:.*]] = fir.alloca i16
// CHECK-NEXT:   %[[V1:.*]] = fir.alloca !fir.array<1xi16>
// CHECK-NEXT:   %[[V2:.*]]:2 = hlfir.declare %arg0 {uniq_name = "_QFtestEarray"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:   %[[V3:.*]]:2 = hlfir.declare %arg2 {uniq_name = "_QFtestEm"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:   %[[V4:.*]]:2 = hlfir.declare %arg1 {uniq_name = "_QFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK-NEXT:   %[[V5:.*]] = fir.load %[[V4]]#0 : !fir.ref<i32>
// CHECK-NEXT:   %[[V6:.*]] = hlfir.designate %[[V1]] (%c1)  : (!fir.ref<!fir.array<1xi16>>, index) -> !fir.ref<i16>
// CHECK-NEXT:   fir.store %c0_i16 to %[[V6]] : !fir.ref<i16>
// CHECK-NEXT:   fir.store %c0_i16 to %[[V0]] : !fir.ref<i16>
// CHECK-NEXT:   %[[V7:.*]]:3 = fir.box_dims %[[V2]]#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK-NEXT:   %[[V8:.*]] = arith.subi %[[V7]]#1, %c1 : index
// CHECK-NEXT:   %[[V9:.*]] = fir.do_loop %arg3 = %c0 to %[[V8]] step %c1 iter_args(%arg4 = %c2147483647_i32) -> (i32) {
// CHECK-NEXT:     %[[V15:.*]] = arith.addi %arg3, %c1 : index
// CHECK-NEXT:     %[[V16:.*]] = hlfir.designate %[[V2]]#0 (%[[V15]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:     %[[V17:.*]] = fir.load %[[V16]] : !fir.ref<i32>
// CHECK-NEXT:     %[[V18:.*]] = arith.cmpi sge, %[[V17]], %[[V5]] : i32
// CHECK-NEXT:     %[[V19:.*]] = fir.if %[[V18]] -> (i32) {
// CHECK-NEXT:       fir.store %c1_i16 to %[[V0]] : !fir.ref<i16>
// CHECK-NEXT:       %[[V20:.*]]:3 = fir.box_dims %[[V2]]#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK-NEXT:       %[[V21:.*]] = arith.subi %[[V20]]#0, %c1 : index
// CHECK-NEXT:       %[[V22:.*]] = arith.addi %[[V15]], %[[V21]] : index
// CHECK-NEXT:       %[[V23:.*]] = hlfir.designate %[[V2]]#0 (%[[V22]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:       %[[V24:.*]] = fir.load %[[V23]] : !fir.ref<i32>
// CHECK-NEXT:       %[[V25:.*]] = arith.cmpi slt, %[[V24]], %arg4 : i32
// CHECK-NEXT:       %[[V26:.*]] = fir.if %[[V25]] -> (i32) {
// CHECK-NEXT:         %[[V27:.*]] = hlfir.designate %[[V1]] (%c1)  : (!fir.ref<!fir.array<1xi16>>, index) -> !fir.ref<i16>
// CHECK-NEXT:         %[[V28:.*]] = fir.convert %[[V15]] : (index) -> i16
// CHECK-NEXT:         fir.store %[[V28]] to %[[V27]] : !fir.ref<i16>
// CHECK-NEXT:         fir.result %[[V24]] : i32
// CHECK-NEXT:       } else {
// CHECK-NEXT:         fir.result %arg4 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       fir.result %[[V26]] : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       fir.result %arg4 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     fir.result %[[V19]] : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[V10:.*]] = fir.load %[[V0]] : !fir.ref<i16>
// CHECK-NEXT:   %[[V11:.*]] = arith.cmpi eq, %[[V10]], %c1_i16 : i16
// CHECK-NEXT:   fir.if %[[V11]] {
// CHECK-NEXT:     %[[V15]] = arith.cmpi eq, %[[V9]], %c2147483647_i32 : i32
// CHECK-NEXT:     fir.if %[[V15]] {
// CHECK-NEXT:       %[[V16]] = hlfir.designate %[[V1]] (%c1)  : (!fir.ref<!fir.array<1xi16>>, index) -> !fir.ref<i16>
// CHECK-NEXT:       fir.store %c1_i16 to %[[V16]] : !fir.ref<i16>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[V12:.*]] = hlfir.as_expr %[[V1]] move %false : (!fir.ref<!fir.array<1xi16>>, i1) -> !hlfir.expr<1xi16>
// CHECK-NEXT:   %[[V13:.*]] = fir.shape %c1 : (index) -> !fir.shape<1>
// CHECK-NEXT:   %[[V14:.*]] = hlfir.elemental %[[V13]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
// CHECK-NEXT:   ^bb0(%arg3: index):
// CHECK-NEXT:     %[[V15:.*]] = hlfir.apply %[[V12]], %arg3 : (!hlfir.expr<1xi16>, index) -> i16
// CHECK-NEXT:     %[[V16:.*]] = fir.convert %[[V15]] : (i16) -> i32
// CHECK-NEXT:     hlfir.yield_element %[[V16]] : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   hlfir.assign %[[V14]] to %[[V3]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:   hlfir.destroy %[[V14]] : !hlfir.expr<?xi32>
// CHECK-NEXT:   return



func.func @_QPtest_float(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "array"}, %arg1: !fir.ref<f32> {fir.bindc_name = "val"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "m"}) {
  %c0 = arith.constant 0 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtestEarray"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFtestEm"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFtestEval"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %3 = fir.load %2#0 : !fir.ref<f32>
  %4:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
  %5 = fir.shape %4#1 : (index) -> !fir.shape<1>
  %6 = hlfir.elemental %5 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%arg3: index):
    %8 = hlfir.designate %0#0 (%arg3)  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %9 = fir.load %8 : !fir.ref<f32>
    %10 = arith.cmpf oge, %9, %3 : f32
    %11 = fir.convert %10 : (i1) -> !fir.logical<4>
    hlfir.yield_element %11 : !fir.logical<4>
  }
  %7 = hlfir.minloc %0#0 mask %6 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xf32>>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  hlfir.assign %7 to %1#0 : !hlfir.expr<1xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %7 : !hlfir.expr<1xi32>
  hlfir.destroy %6 : !hlfir.expr<?x!fir.logical<4>>
  return
}
// CHECK-LABEL: _QPtest_float
// CHECK:        %[[V11:.*]] = fir.do_loop %arg3 = %c0 to %[[V10:.*]] step %c1 iter_args(%arg4 = %cst) -> (f32) {
// CHECK-NEXT:     %[[V14:.*]] = arith.addi %arg3, %c1 : index
// CHECK-NEXT:     %[[V15:.*]] = hlfir.designate %[[V1:.*]]#0 (%[[V14]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK-NEXT:     %[[V16:.*]] = fir.load %[[V15]] : !fir.ref<f32>
// CHECK-NEXT:     %[[V17:.*]] = arith.cmpf oge, %[[V16]], %[[V4:.*]] : f32
// CHECK-NEXT:     %[[V18:.*]] = fir.if %[[V17]] -> (f32) {
// CHECK-NEXT:       fir.store %c1_i32 to %[[V0:.*]] : !fir.ref<i32>
// CHECK-NEXT:       %[[DIMS:.*]]:3 = fir.box_dims %2#0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
// CHECK-NEXT:       %[[SUB:.*]] = arith.subi %[[DIMS]]#0, %c1 : index
// CHECK-NEXT:       %[[ADD:.*]] = arith.addi %[[V14]], %[[SUB]] : index
// CHECK-NEXT:       %[[V19:.*]] = hlfir.designate %[[V1]]#0 (%[[ADD]]) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK-NEXT:       %[[V20:.*]] = fir.load %[[V19]] : !fir.ref<f32>
// CHECK-NEXT:       %[[V21:.*]] = arith.cmpf olt, %[[V20]], %arg4 fastmath<contract> : f32
// CHECK-NEXT:       %[[V22:.*]] = fir.if %[[V21]] -> (f32) {
// CHECK-NEXT:         %[[V23:.*]] = hlfir.designate %{{.}} (%c1) : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:         %[[V24:.*]] = fir.convert %[[V14]] : (index) -> i32
// CHECK-NEXT:         fir.store %[[V24]] to %[[V23]] : !fir.ref<i32>
// CHECK-NEXT:         fir.result %[[V20]] : f32
// CHECK-NEXT:       } else {
// CHECK-NEXT:         fir.result %arg4 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:       fir.result %[[V22]] : f32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       fir.result %arg4 : f32
// CHECK-NEXT:     }
// CHECK-NEXT:     fir.result %[[V18]] : f32
// CHECK-NEXT:   }


func.func @_QPtest_assignshape(%arg0: !fir.ref<!fir.array<3x3xf32>> {fir.bindc_name = "array"}, %arg1: !fir.ref<f32> {fir.bindc_name = "val"}, %arg2: !fir.ref<!fir.array<3xi32>> {fir.bindc_name = "m"}) {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.shape %c3, %c3 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFtestEarray"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
  %2 = fir.shape %c3 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg2(%2) {uniq_name = "_QFtestEm"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
  %4:2 = hlfir.declare %arg1 {uniq_name = "_QFtestEval"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %5 = fir.load %4#0 : !fir.ref<f32>
  %6 = hlfir.elemental %0 unordered : (!fir.shape<2>) -> !hlfir.expr<3x3x!fir.logical<4>> {
  ^bb0(%arg3: index, %arg4: index):
    %10 = hlfir.designate %1#0 (%arg3, %arg4)  : (!fir.ref<!fir.array<3x3xf32>>, index, index) -> !fir.ref<f32>
    %11 = fir.load %10 : !fir.ref<f32>
    %12 = arith.cmpf oge, %11, %5 : f32
    %13 = fir.convert %12 : (i1) -> !fir.logical<4>
    hlfir.yield_element %13 : !fir.logical<4>
  }
  %7 = hlfir.minloc %1#0 mask %6 {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<3x3xf32>>, !hlfir.expr<3x3x!fir.logical<4>>) -> !hlfir.expr<2xi32>
  %8 = fir.shape %c2 : (index) -> !fir.shape<1>
  %9 = hlfir.designate %3#0 (%c1:%c2:%c1)  shape %8 : (!fir.ref<!fir.array<3xi32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<2xi32>>
  hlfir.assign %7 to %9 : !hlfir.expr<2xi32>, !fir.ref<!fir.array<2xi32>>
  hlfir.destroy %7 : !hlfir.expr<2xi32>
  hlfir.destroy %6 : !hlfir.expr<3x3x!fir.logical<4>>
  return
}
// Not supported as the input is not a box
// CHECK-LABEL: _QPtest_assignshape
// CHECK: hlfir.minloc


func.func @_QFPtest_character(%arg0: !fir.box<!fir.array<?x!fir.char<1>>> {fir.bindc_name = "b"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "c"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0:2 = hlfir.declare %arg0 typeparams %c1 {uniq_name = "_QFFtestEb"} : (!fir.box<!fir.array<?x!fir.char<1>>>, index) -> (!fir.box<!fir.array<?x!fir.char<1>>>, !fir.box<!fir.array<?x!fir.char<1>>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFtestEc"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2 = fir.alloca !fir.array<1xi32> {bindc_name = "m", uniq_name = "_QFFtestEm"}
  %3 = fir.shape %c1 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %2(%3) {uniq_name = "_QFFtestEm"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
  %5 = fir.alloca i32 {bindc_name = "test", uniq_name = "_QFFtestEtest"}
  %6:2 = hlfir.declare %5 {uniq_name = "_QFFtestEtest"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7:2 = hlfir.declare %arg2 {uniq_name = "_QFFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %8 = fir.load %7#0 : !fir.ref<i32>
  %9:3 = fir.box_dims %1#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %10 = fir.shape %9#1 : (index) -> !fir.shape<1>
  %11 = hlfir.elemental %10 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%arg3: index):
    %16 = hlfir.designate %1#0 (%arg3)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %17 = fir.load %16 : !fir.ref<i32>
    %18 = arith.cmpi eq, %17, %8 : i32
    %19 = fir.convert %18 : (i1) -> !fir.logical<4>
    hlfir.yield_element %19 : !fir.logical<4>
  }
  %12 = hlfir.minloc %0#0 mask %11 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.char<1>>>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  hlfir.assign %12 to %4#0 : !hlfir.expr<1xi32>, !fir.ref<!fir.array<1xi32>>
  hlfir.destroy %12 : !hlfir.expr<1xi32>
  hlfir.destroy %11 : !hlfir.expr<?x!fir.logical<4>>
  %13 = hlfir.designate %4#0 (%c1)  : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
  %14 = fir.load %13 : !fir.ref<i32>
  hlfir.assign %14 to %6#0 : i32, !fir.ref<i32>
  %15 = fir.load %6#1 : !fir.ref<i32>
  return %15 : i32
}
// Characters are not supported at the moment
// CHECK-LABEL: _QFPtest_character
// CHECK: hlfir.minloc


func.func @_QPtest_parts(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "mask"}) -> f32 {
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c0 = arith.constant 0 : index
  %c5_i32 = arith.constant 5 : i32
  %0:2 = hlfir.declare %arg1 {uniq_name = "_QFtestEmask"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1 = fir.alloca f32 {bindc_name = "test", uniq_name = "_QFtestEtest"}
  %2:2 = hlfir.declare %1 {uniq_name = "_QFtestEtest"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %3:2 = hlfir.declare %arg0 {uniq_name = "_QFtestEx"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %4:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %5 = fir.shape %4#1 : (index) -> !fir.shape<1>
  %6 = hlfir.elemental %5 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%arg2: index):
    %11 = hlfir.designate %0#0 (%arg2)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %12 = fir.load %11 : !fir.ref<i32>
    %13 = arith.cmpi sge, %12, %c5_i32 : i32
    %14 = fir.convert %13 : (i1) -> !fir.logical<4>
    hlfir.yield_element %14 : !fir.logical<4>
  }
  %7 = hlfir.minloc %3#0 mask %6 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  %8 = fir.shape %c1 : (index) -> !fir.shape<1>
  %9 = hlfir.designate %3#0 (%c5:%c5:%c1)  shape %8 : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
  hlfir.assign %7 to %9 : !hlfir.expr<1xi32>, !fir.box<!fir.array<1xi32>>
  hlfir.destroy %7 : !hlfir.expr<1xi32>
  hlfir.destroy %6 : !hlfir.expr<?x!fir.logical<4>>
  %10 = fir.load %2#1 : !fir.ref<f32>
  return %10 : f32
}
// Characters are not supported at the moment
// CHECK-LABEL: _QPtest_parts
// CHECK: fir.do_loop %{{.*}} = %c0 to %{{.*}} step %c1 iter_args(%{{.*}} = %c2147483647_i32) -> (i32) {

