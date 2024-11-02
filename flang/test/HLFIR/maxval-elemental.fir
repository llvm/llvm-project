// Test maxval inlining for both elemental and designate
// RUN: fir-opt %s -opt-bufferization | FileCheck %s

// subroutine test(array)
//   integer :: array(:), x
//   x = maxval(abs(array))
// end subroutine test

func.func @_QPtest(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"}) {
  %c31_i32 = arith.constant 31 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtestEarray"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2 = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtestEx"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFtestEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:3 = fir.box_dims %1#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %5 = fir.shape %4#1 : (index) -> !fir.shape<1>
  %6 = hlfir.elemental %5 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg1: index):
    %8 = hlfir.designate %1#0 (%arg1)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %9 = fir.load %8 : !fir.ref<i32>
    %10 = arith.shrsi %9, %c31_i32 : i32
    %11 = arith.xori %9, %10 : i32
    %12 = arith.subi %11, %10 : i32
    hlfir.yield_element %12 : i32
  }
  %7 = hlfir.maxval %6 {fastmath = #arith.fastmath<contract>} : (!hlfir.expr<?xi32>) -> i32
  hlfir.assign %7 to %3#0 : i32, !fir.ref<i32>
  hlfir.destroy %6 : !hlfir.expr<?xi32>
  return
}

// CHECK-LABEL: func.func @_QPtest(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "array"}) {
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c-2147483648_i32 = arith.constant -2147483648 : i32
// CHECK-NEXT:    %c31_i32 = arith.constant 31 : i32
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %[[V0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK-NEXT:    %[[V1:.*]]:2 = hlfir.declare %arg0 dummy_scope %[[V0]] {uniq_name = "_QFtestEarray"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %[[V2:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtestEx"}
// CHECK-NEXT:    %[[V3:.*]]:2 = hlfir.declare %[[V2]] {uniq_name = "_QFtestEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK-NEXT:    %[[V4:.*]]:3 = fir.box_dims %[[V1]]#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK-NEXT:    %[[V5:.*]] = fir.do_loop %arg1 = %c1 to %[[V4]]#1 step %c1 iter_args(%arg2 = %c-2147483648_i32) -> (i32) {
// CHECK-NEXT:      %[[V6:.*]] = hlfir.designate %[[V1]]#0 (%arg1)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:      %[[V7:.*]] = fir.load %[[V6]] : !fir.ref<i32>
// CHECK-NEXT:      %[[V8:.*]] = arith.shrsi %[[V7]], %c31_i32 : i32
// CHECK-NEXT:      %[[V9:.*]] = arith.xori %[[V7]], %[[V8]] : i32
// CHECK-NEXT:      %[[V10:.*]] = arith.subi %[[V9]], %[[V8]] : i32
// CHECK-NEXT:      %[[V11:.*]] = arith.cmpi sgt, %[[V10]], %arg2 : i32
// CHECK-NEXT:      %[[V12:.*]] = arith.select %[[V11]], %[[V10]], %arg2 : i32
// CHECK-NEXT:      fir.result %[[V12]] : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    hlfir.assign %[[V5]] to %[[V3]]#0 : i32, !fir.ref<i32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// subroutine test(array)
//   real :: array(:), x
//   x = maxval(array(3:6))
// end subroutine test

func.func @_QPtest_float(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "array"}) {
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c3 = arith.constant 3 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtestEarray"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %2 = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFtestEx"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFtestEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %4 = fir.shape %c4 : (index) -> !fir.shape<1>
  %5 = hlfir.designate %1#0 (%c3:%c6:%c1)  shape %4 : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<4xf32>>
  %6 = hlfir.maxval %5 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<4xf32>>) -> f32
  hlfir.assign %6 to %3#0 : f32, !fir.ref<f32>
  return
}

// CHECK-LABEL: _QPtest_float
// CHECK:       %cst = arith.constant 0xFF800000 : f32
// CHECK:       %[[V4:.*]] = fir.shape %c4 : (index) -> !fir.shape<1>
// CHECK-NEXT:  %[[V5:.*]] = hlfir.designate %{{.*}} (%c3:%c6:%c1)  shape %[[V4]] : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<4xf32>>
// CHECK-NEXT:  %[[V6:.*]] = fir.do_loop %arg1 = %c1 to %c4 step %c1 iter_args(%arg2 = %cst) -> (f32) {
// CHECK-NEXT:      %[[V7:.*]] = hlfir.designate %[[V5]] (%arg1)  : (!fir.box<!fir.array<4xf32>>, index) -> !fir.ref<f32>
// CHECK-NEXT:      %[[V8:.*]] = fir.load %[[V7]] : !fir.ref<f32>
// CHECK-NEXT:      %[[V9:.*]] = arith.cmpf ogt, %[[V8]], %arg2 : f32
// CHECK-NEXT:      %[[V10:.*]] = arith.cmpf une, %arg2, %arg2 : f32
// CHECK-NEXT:      %[[V11:.*]] = arith.cmpf oeq, %[[V8]], %[[V8]] : f32
// CHECK-NEXT:      %[[V12:.*]] = arith.andi %[[V10]], %[[V11]] : i1
// CHECK-NEXT:      %[[V13:.*]] = arith.ori %[[V9]], %[[V12]] : i1
// CHECK-NEXT:      %[[V14:.*]] = arith.select %[[V13]], %[[V8]], %arg2 : f32
// CHECK-NEXT:      fir.result %[[V14]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    hlfir.assign %[[V6]] to %3#0 : f32, !fir.ref<f32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
