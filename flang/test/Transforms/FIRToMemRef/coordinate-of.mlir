// Tests for fir.coordinate_of lowering in FIRToMemRef.
//
// Accepted conversions: 1-D (block-argument base), 1-D (store),
//   1-D (alloca-defined base via getFIRConvert), 2-D and 3-D (shape and
//   index reversal from FIR column-major to memref row-major).
// Rejected conversions: dynamic-extent array (hasDynamicSize → falls back
//   to rank-0 scalar memref; fir.coordinate_of is NOT erased).
//
// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// ----------------------------------------------------------------------------
// Accepted: 1-D load, block-argument base (fs28932c pattern).
// The base is a region/function block argument so it is converted directly
// via fir.convert; fir.coordinate_of is erased.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @coor_1d_load_block_arg
// CHECK:       [[BASE:%.+]] = fir.convert %arg0 : (!fir.ref<!fir.array<3xf32>>) -> memref<3xf32>
// CHECK:       memref.load [[BASE]][%c1] : memref<3xf32>
// CHECK-NOT:   fir.coordinate_of
func.func @coor_1d_load_block_arg(%arg0: !fir.ref<!fir.array<3xf32>>) {
  %c1 = arith.constant 1 : index
  %ptr = fir.coordinate_of %arg0, %c1 : (!fir.ref<!fir.array<3xf32>>, index) -> !fir.ref<f32>
  %val = fir.load %ptr : !fir.ref<f32>
  return
}

// ----------------------------------------------------------------------------
// Accepted: 1-D store, block-argument base.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @coor_1d_store_block_arg
// CHECK:       [[BASE:%.+]] = fir.convert %arg0 : (!fir.ref<!fir.array<3xf32>>) -> memref<3xf32>
// CHECK:       memref.store {{%.+}}, [[BASE]][%c0] : memref<3xf32>
// CHECK-NOT:   fir.coordinate_of
func.func @coor_1d_store_block_arg(%arg0: !fir.ref<!fir.array<3xf32>>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.0 : f32
  %ptr = fir.coordinate_of %arg0, %c0 : (!fir.ref<!fir.array<3xf32>>, index) -> !fir.ref<f32>
  fir.store %cst to %ptr : !fir.ref<f32>
  return
}

// ----------------------------------------------------------------------------
// Accepted: 1-D load, alloca-defined base (getFIRConvert path).
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @coor_1d_load_alloca
// CHECK:       memref.alloca() : memref<3xf32>
// CHECK:       memref.load {{%.+}}[%c2] : memref<3xf32>
// CHECK-NOT:   fir.coordinate_of
func.func @coor_1d_load_alloca() {
  %c2 = arith.constant 2 : index
  %alloc = fir.alloca !fir.array<3xf32>
  %ptr = fir.coordinate_of %alloc, %c2 : (!fir.ref<!fir.array<3xf32>>, index) -> !fir.ref<f32>
  %val = fir.load %ptr : !fir.ref<f32>
  return
}

// ----------------------------------------------------------------------------
// Accepted: 2-D load — shape and index reversal.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @coor_2d_load
// CHECK:       [[BASE:%.+]] = fir.convert %arg0 : (!fir.ref<!fir.array<2x3xf32>>) -> memref<3x2xf32>
// CHECK:       memref.load [[BASE]][%arg2, %arg1] : memref<3x2xf32>
// CHECK-NOT:   fir.coordinate_of
func.func @coor_2d_load(%arg0: !fir.ref<!fir.array<2x3xf32>>, %arg1: index, %arg2: index) {
  %ptr = fir.coordinate_of %arg0, %arg1, %arg2
      : (!fir.ref<!fir.array<2x3xf32>>, index, index) -> !fir.ref<f32>
  %val = fir.load %ptr : !fir.ref<f32>
  return
}

// ----------------------------------------------------------------------------
// Accepted: 3-D load — shape and index reversal.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @coor_3d_load
// CHECK:       [[BASE:%.+]] = fir.convert %arg0 : (!fir.ref<!fir.array<2x3x4xf32>>) -> memref<4x3x2xf32>
// CHECK:       memref.load [[BASE]][%arg3, %arg2, %arg1] : memref<4x3x2xf32>
// CHECK-NOT:   fir.coordinate_of
func.func @coor_3d_load(%arg0: !fir.ref<!fir.array<2x3x4xf32>>,
                        %arg1: index, %arg2: index, %arg3: index) {
  %ptr = fir.coordinate_of %arg0, %arg1, %arg2, %arg3
      : (!fir.ref<!fir.array<2x3x4xf32>>, index, index, index) -> !fir.ref<f32>
  %val = fir.load %ptr : !fir.ref<f32>
  return
}

// ----------------------------------------------------------------------------
// Rejected: dynamic-extent array.
// ----------------------------------------------------------------------------
// CHECK-LABEL: func.func @coor_dynamic_reject
// CHECK:       [[COOR:%.+]] = fir.coordinate_of %arg0, %arg1
// CHECK:       [[CONV:%.+]] = fir.convert [[COOR]] : (!fir.ref<f32>) -> memref<f32>
// CHECK:       memref.load [[CONV]][] : memref<f32>
func.func @coor_dynamic_reject(%arg0: !fir.ref<!fir.array<?xf32>>, %arg1: index) {
  %ptr = fir.coordinate_of %arg0, %arg1 : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  %val = fir.load %ptr : !fir.ref<f32>
  return
}
