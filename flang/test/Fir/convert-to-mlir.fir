// RUN: fir-opt --split-input-file --fir-to-mlir %s | FileCheck %s

//===================================================
// SUMMARY: Tests for FIR --> MLIR core dialects conversion
//===================================================

// Test `fir.load` --> `memref.load` conversion

func.func @test_load_f32(%addr : !fir.ref<f32>) -> f32 {
  %0 = fir.load %addr : !fir.ref<f32>
  return %0 : f32
}

// CHECK-LABEL:   func.func @test_load_f32(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<f32>) -> f32 {
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[ARG0]] : (!fir.ref<f32>) -> memref<f32, strided<[], offset: ?>>
// CHECK:           %[[LOAD_0:.*]] = memref.load %[[CONVERT_0]][] : memref<f32, strided<[], offset: ?>>
// CHECK:           return %[[LOAD_0]] : f32
// CHECK:         }

// -----

// Test `fir.store` --> `memref.store` conversion

func.func @test_store_f32(%val : f32, %addr : !fir.ref<f32>) {
  fir.store %val to %addr : !fir.ref<f32>
  return
}

// CHECK-LABEL:   func.func @test_store_f32(
// CHECK-SAME:      %[[ARG0:.*]]: f32,
// CHECK-SAME:      %[[ARG1:.*]]: !fir.ref<f32>) {
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[ARG1]] : (!fir.ref<f32>) -> memref<f32, strided<[], offset: ?>>
// CHECK:           memref.store %[[ARG0]], %[[CONVERT_0]][] : memref<f32, strided<[], offset: ?>>
// CHECK:           return
// CHECK:         }

// -----

// Test `fir.convert` operation conversion between Interger and Index type.

func.func @convert_between_int_and_index(%arg0 : i32) -> i64 {
  %0 = fir.convert %arg0 : (i32) -> index
  %1 = fir.convert %0 : (index) -> i64
  return %1 : i64
}

// CHECK-LABEL:   func.func @convert_between_int_and_index(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i64 {
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG0]] : i32 to index
// CHECK:           %[[INDEX_CAST_1:.*]] = arith.index_cast %[[INDEX_CAST_0]] : index to i64
// CHECK:           return %[[INDEX_CAST_1]] : i64
// CHECK:         }

// -----

// Test `fir.convert` operation conversion between Interger type.

func.func @convert_between_int(%arg0 : i32) -> i16 {
  %0 = fir.convert %arg0 : (i32) -> i64
  %1 = fir.convert %0 : (i64) -> i16
  return %1 : i16
}

// CHECK-LABEL:   func.func @convert_between_int(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i16 {
// CHECK:           %[[EXTSI_0:.*]] = arith.extsi %[[ARG0]] : i32 to i64
// CHECK:           %[[TRUNCI_0:.*]] = arith.trunci %[[EXTSI_0]] : i64 to i16
// CHECK:           return %[[TRUNCI_0]] : i16
// CHECK:         }

// -----

// Test `fir.convert` operation conversion between Float type.

func.func @convert_between_fp(%arg0 : f32) -> f16 {
  %0 = fir.convert %arg0 : (f32) -> f64
  %1 = fir.convert %0 : (f64) -> f16
  return %1 : f16
}

// CHECK-LABEL:   func.func @convert_between_fp(
// CHECK-SAME:      %[[ARG0:.*]]: f32) -> f16 {
// CHECK:           %[[EXTF_0:.*]] = arith.extf %[[ARG0]] : f32 to f64
// CHECK:           %[[TRUNCF_0:.*]] = arith.truncf %[[EXTF_0]] : f64 to f16
// CHECK:           return %[[TRUNCF_0]] : f16
// CHECK:         }

// -----

// Test `fir.alloca` --> `memref.alloca` conversion

func.func @test_alloca_f32() -> !fir.ref<f32> {
  %1 = fir.alloca f32
  return %1 : !fir.ref<f32>
}

// CHECK-LABEL:   func.func @test_alloca_f32() -> !fir.ref<f32> {
// CHECK:           %[[ALLOCA_0:.*]] = memref.alloca() {in_type = f32} : memref<f32>
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[ALLOCA_0]] : memref<f32> to memref<f32, strided<[], offset: ?>>
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[CAST_0]] : (memref<f32, strided<[], offset: ?>>) -> !fir.ref<f32>
// CHECK:           return %[[CONVERT_0]] : !fir.ref<f32>
// CHECK:         }

// -----

// Test `fircg.ext_array_coor` conversion.

func.func @test_ext_array_coor(%arg0: !fir.ref<!fir.array<100x200xf32>>, %i : i64, %j : i64) -> !fir.ref<f32> {
  %c200 = arith.constant 200 : index
  %c100 = arith.constant 100 : index
  %0 = fircg.ext_array_coor %arg0(%c100, %c200)<%i, %j> : (!fir.ref<!fir.array<100x200xf32>>, index, index, i64, i64) -> !fir.ref<f32>
  return %0 : !fir.ref<f32>
}

// CHECK-LABEL:   func.func @test_ext_array_coor(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.array<100x200xf32>>,
// CHECK-SAME:      %[[ARG1:.*]]: i64,
// CHECK-SAME:      %[[ARG2:.*]]: i64) -> !fir.ref<f32> {
// CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[ARG0]] : (!fir.ref<!fir.array<100x200xf32>>) -> memref<f32, strided<[], offset: ?>>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 200 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_0:.*]], %[[EXTRACT_STRIDED_METADATA_0:.*]] = memref.extract_strided_metadata %[[CONVERT_0]] : memref<f32, strided<[], offset: ?>> -> memref<f32>, index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_2]], %[[CONSTANT_1]] : index
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[EXTRACT_STRIDED_METADATA_0]]], sizes: {{\[}}%[[CONSTANT_0]], %[[CONSTANT_1]]], strides: {{\[}}%[[MULI_0]], 1] : memref<f32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG2]] : i64 to index
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[INDEX_CAST_0]], %[[CONSTANT_3]] : index
// CHECK:           %[[INDEX_CAST_1:.*]] = arith.index_cast %[[ARG1]] : i64 to index
// CHECK:           %[[SUBI_1:.*]] = arith.subi %[[INDEX_CAST_1]], %[[CONSTANT_3]] : index
// CHECK:           %[[SUBVIEW_0:.*]] = memref.subview %[[REINTERPRET_CAST_0]]{{\[}}%[[SUBI_0]], %[[SUBI_1]]] [1, 1] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
// CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[SUBVIEW_0]] : (memref<f32, strided<[], offset: ?>>) -> !fir.ref<f32>
// CHECK:           return %[[CONVERT_1]] : !fir.ref<f32>
// CHECK:         }
