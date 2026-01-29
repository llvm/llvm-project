/// Verify that converts are only generated correctly without declare ops

// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL:   func.func @nodeclare
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[SHAPE:.*]] = fir.shape %[[C1]] : (index) -> !fir.shape<1>
// CHECK:         %[[COOR:.*]] = fir.array_coor %arg0(%[[SHAPE]]) %[[C1]] : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:         %[[C0:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> memref<i32>
// CHECK:         %[[C1M:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> memref<i32>
// CHECK:         %[[L0:.*]] = memref.load %[[C1M]][] : memref<i32>
// CHECK:         %[[CARG1:.*]] = fir.convert %arg1 : (!fir.ref<i32>) -> memref<i32>
// CHECK:         memref.store %[[L0]], %[[CARG1]][] : memref<i32>
// CHECK:         %[[L1:.*]] = memref.load %[[C0]][] : memref<i32>
// CHECK:         %[[CARG2:.*]] = fir.convert %arg2 : (!fir.ref<i32>) -> memref<i32>
// CHECK:         memref.store %[[L1]], %[[CARG2]][] : memref<i32>

func.func @nodeclare(%arg0: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "b"}, %arg2: !fir.ref<i32> {fir.bindc_name = "c"}) attributes {fir.internal_name = ""} {
  %c1 = arith.constant 1 : index
  %shape = fir.shape %c1 : (index) -> !fir.shape<1>
  %0 = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  %1 = fir.load %0 : !fir.ref<i32>
  fir.store %1 to %arg1 : !fir.ref<i32>
  %2 = fir.load %0 : !fir.ref<i32>
  fir.store %2 to %arg2 : !fir.ref<i32>
  return
}

// CHECK-LABEL:   func.func @nodeclare_regions
// CHECK-COUNT-4: fir.convert %{{.*}} : (!fir.ref<i32>) -> memref<i32>
// CHECK-COUNT-1: fir.convert %{{.*}} : (i32) -> f32

func.func @nodeclare_regions(%arg0: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "h11"}, %arg1: !fir.ref<!fir.array<6xi32>> {fir.bindc_name = "rslt"}) attributes {fir.internal_name = "_QPsub11"} {
  %cst = arith.constant 1.100000e+01 : f32
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c6 = arith.constant 6 : index
  %shape = fir.shape %c6 : (index) -> !fir.shape<1>
  %0 = fir.array_coor %arg1(%shape) %c1 : (!fir.ref<!fir.array<6xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  fir.store %c0_i32 to %0 : !fir.ref<i32>
  %1 = fir.load %0 : !fir.ref<i32>
  %2 = fir.convert %1 : (i32) -> f32
  %3 = arith.cmpf une, %2, %cst fastmath<contract> : f32
  fir.if %3 {
    %4 = fir.load %0 : !fir.ref<i32>
    %5 = arith.addi %4, %c1_i32 : i32
    fir.store %5 to %0 : !fir.ref<i32>
  }
  return
}
