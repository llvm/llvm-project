// RUN: fir-opt --split-input-file --fir-to-core-mlir %s | FileCheck %s

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
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !fir.ref<f32> to memref<f32, strided<[], offset: ?>>
// CHECK:           %[[LOAD_0:.*]] = memref.load %[[UNREALIZED_CONVERSION_CAST_0]][] : memref<f32, strided<[], offset: ?>>
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
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !fir.ref<f32> to memref<f32, strided<[], offset: ?>>
// CHECK:           memref.store %[[ARG0]], %[[UNREALIZED_CONVERSION_CAST_0]][] : memref<f32, strided<[], offset: ?>>
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
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[CAST_0]] : memref<f32, strided<[], offset: ?>> to !fir.ref<f32>
// CHECK:           return %[[UNREALIZED_CONVERSION_CAST_0]] : !fir.ref<f32>
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
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !fir.ref<!fir.array<100x200xf32>> to memref<f32, strided<[], offset: ?>>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 200 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_0:.*]], %[[EXTRACT_STRIDED_METADATA_0:.*]] = memref.extract_strided_metadata %[[UNREALIZED_CONVERSION_CAST_0]] : memref<f32, strided<[], offset: ?>> -> memref<f32>, index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[CONSTANT_2]], %[[CONSTANT_1]] : index
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[EXTRACT_STRIDED_METADATA_0]]], sizes: {{\[}}%[[CONSTANT_0]], %[[CONSTANT_1]]], strides: {{\[}}%[[MULI_0]], 1] : memref<f32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG2]] : i64 to index
// CHECK:           %[[SUBI_0:.*]] = arith.subi %[[INDEX_CAST_0]], %[[CONSTANT_3]] : index
// CHECK:           %[[INDEX_CAST_1:.*]] = arith.index_cast %[[ARG1]] : i64 to index
// CHECK:           %[[SUBI_1:.*]] = arith.subi %[[INDEX_CAST_1]], %[[CONSTANT_3]] : index
// CHECK:           %[[SUBVIEW_0:.*]] = memref.subview %[[REINTERPRET_CAST_0]]{{\[}}%[[SUBI_0]], %[[SUBI_1]]] [1, 1] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[SUBVIEW_0]] : memref<f32, strided<[], offset: ?>> to !fir.ref<f32>
// CHECK:           return %[[UNREALIZED_CONVERSION_CAST_1]] : !fir.ref<f32>
// CHECK:         }

// -----

// Test `fir.select_case` operation conversion with INTEGER.

func.func @select_case_integer(%arg0: memref<i32>) -> i32 {
  %2 = memref.load %arg0[] : memref<i32>
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %c5_i32 = arith.constant 5 : i32
  %c7_i32 = arith.constant 7 : i32
  %c8_i32 = arith.constant 8 : i32
  %c15_i32 = arith.constant 15 : i32
  %c21_i32 = arith.constant 21 : i32
  fir.select_case %2 : i32 [#fir.upper, %c1_i32, ^bb1,
                            #fir.point, %c2_i32, ^bb2,
                            #fir.interval, %c4_i32, %c5_i32, ^bb4,
                            #fir.point, %c7_i32, ^bb5,
                            #fir.interval, %c8_i32, %c15_i32, ^bb5,
                            #fir.lower, %c21_i32, ^bb5,
                            unit, ^bb3]
^bb1:  // pred: ^bb0
  %c1_i32_0 = arith.constant 1 : i32
  memref.store %c1_i32_0, %arg0[] : memref<i32>
  cf.br ^bb6
^bb2:  // pred: ^bb0
  %c2_i32_1 = arith.constant 2 : i32
  memref.store %c2_i32_1, %arg0[] : memref<i32>
  cf.br ^bb6
^bb3:  // pred: ^bb0
  %c0_i32 = arith.constant 0 : i32
  memref.store %c0_i32, %arg0[] : memref<i32>
  cf.br ^bb6
^bb4:  // pred: ^bb0
  %c4_i32_2 = arith.constant 4 : i32
  memref.store %c4_i32_2, %arg0[] : memref<i32>
  cf.br ^bb6
^bb5:  // 3 preds: ^bb0, ^bb0, ^bb0
  %c7_i32_3 = arith.constant 7 : i32
  memref.store %c7_i32_3, %arg0[] : memref<i32>
  cf.br ^bb6
^bb6:  // 5 preds: ^bb1, ^bb2, ^bb3, ^bb4, ^bb5
  %3 = memref.load %arg0[] : memref<i32>
  return %3 : i32
}

// CHECK-LABEL:   func.func @select_case_integer(
// CHECK-SAME:      %[[ARG0:.*]]: memref<i32>) -> i32 {
// CHECK:           %[[LOAD_0:.*]] = memref.load %[[ARG0]][] : memref<i32>
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 5 : i32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 7 : i32
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant 8 : i32
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 15 : i32
// CHECK:           %[[CONSTANT_7:.*]] = arith.constant 21 : i32
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi sle, %[[LOAD_0]], %[[CONSTANT_0]] : i32
// CHECK:           cf.cond_br %[[CMPI_0]], ^bb7, ^bb1
// CHECK:         ^bb1:
// CHECK:           %[[CMPI_1:.*]] = arith.cmpi eq, %[[LOAD_0]], %[[CONSTANT_1]] : i32
// CHECK:           cf.cond_br %[[CMPI_1]], ^bb8, ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[CMPI_2:.*]] = arith.cmpi sge, %[[LOAD_0]], %[[CONSTANT_2]] : i32
// CHECK:           %[[CMPI_3:.*]] = arith.cmpi sle, %[[LOAD_0]], %[[CONSTANT_3]] : i32
// CHECK:           %[[ANDI_0:.*]] = arith.andi %[[CMPI_2]], %[[CMPI_3]] : i1
// CHECK:           cf.cond_br %[[ANDI_0]], ^bb10, ^bb3
// CHECK:         ^bb3:
// CHECK:           %[[CMPI_4:.*]] = arith.cmpi eq, %[[LOAD_0]], %[[CONSTANT_4]] : i32
// CHECK:           cf.cond_br %[[CMPI_4]], ^bb11, ^bb4
// CHECK:         ^bb4:
// CHECK:           %[[CMPI_5:.*]] = arith.cmpi sge, %[[LOAD_0]], %[[CONSTANT_5]] : i32
// CHECK:           %[[CMPI_6:.*]] = arith.cmpi sle, %[[LOAD_0]], %[[CONSTANT_6]] : i32
// CHECK:           %[[ANDI_1:.*]] = arith.andi %[[CMPI_5]], %[[CMPI_6]] : i1
// CHECK:           cf.cond_br %[[ANDI_1]], ^bb11, ^bb5
// CHECK:         ^bb5:
// CHECK:           %[[CMPI_7:.*]] = arith.cmpi sge, %[[LOAD_0]], %[[CONSTANT_7]] : i32
// CHECK:           cf.cond_br %[[CMPI_7]], ^bb11, ^bb6
// CHECK:         ^bb6:
// CHECK:           cf.br ^bb9
// CHECK:         ^bb7:
// CHECK:           %[[CONSTANT_8:.*]] = arith.constant 1 : i32
// CHECK:           memref.store %[[CONSTANT_8]], %[[ARG0]][] : memref<i32>
// CHECK:           cf.br ^bb12
// CHECK:         ^bb8:
// CHECK:           %[[CONSTANT_9:.*]] = arith.constant 2 : i32
// CHECK:           memref.store %[[CONSTANT_9]], %[[ARG0]][] : memref<i32>
// CHECK:           cf.br ^bb12
// CHECK:         ^bb9:
// CHECK:           %[[CONSTANT_10:.*]] = arith.constant 0 : i32
// CHECK:           memref.store %[[CONSTANT_10]], %[[ARG0]][] : memref<i32>
// CHECK:           cf.br ^bb12
// CHECK:         ^bb10:
// CHECK:           %[[CONSTANT_11:.*]] = arith.constant 4 : i32
// CHECK:           memref.store %[[CONSTANT_11]], %[[ARG0]][] : memref<i32>
// CHECK:           cf.br ^bb12
// CHECK:         ^bb11:
// CHECK:           %[[CONSTANT_12:.*]] = arith.constant 7 : i32
// CHECK:           memref.store %[[CONSTANT_12]], %[[ARG0]][] : memref<i32>
// CHECK:           cf.br ^bb12
// CHECK:         ^bb12:
// CHECK:           %[[LOAD_1:.*]] = memref.load %[[ARG0]][] : memref<i32>
// CHECK:           return %[[LOAD_1]] : i32
// CHECK:         }
