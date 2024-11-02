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
  %7 = hlfir.maxloc %0#0 mask %6 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
  hlfir.assign %7 to %1#0 : !hlfir.expr<1xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %7 : !hlfir.expr<1xi32>
  hlfir.destroy %6 : !hlfir.expr<?x!fir.logical<4>>
  return
}
// CHECK-LABEL: func.func @_QPtest(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"}, %arg1: !fir.ref<i32> {fir.bindc_name = "val"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "m"}) {
// CHECK-NEXT:    %c-2147483648_i32 = arith.constant -2147483648 : i32
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
// CHECK-NEXT:    %[[V11:.*]] = fir.do_loop %arg3 = %c0 to %[[V10]] step %c1 iter_args(%arg4 = %c-2147483648_i32) -> (i32) {
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
// CHECK-NEXT:        %[[V21:.*]] = arith.cmpi sgt, %[[V20]], %arg4 : i32
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
// CHECK-NEXT:      %[[V14:.*]] = arith.cmpi eq, %[[V11]], %c-2147483648_i32 : i32
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
  %7 = hlfir.maxloc %0#0 mask %6 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xf32>>, !hlfir.expr<?x!fir.logical<4>>) -> !hlfir.expr<1xi32>
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
// CHECK-NEXT:       %[[V21:.*]] = arith.cmpf ogt, %[[V20]], %arg4 fastmath<contract> : f32
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

