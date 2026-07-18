// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" --split-input-file %s | FileCheck %s

// This test ensures that the FIR CodeGen ConvertOpConversion
// properly lowers fir.convert when either the source or the destination
// type is a memref.

// CHECK-LABEL: llvm.func @memref_to_ref_convert(
// Reconstruct the memref descriptor from the expanded LLVM arguments.
// CHECK:         %[[POISON0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
// CHECK:         %[[DESC0:.*]] = llvm.insertvalue %arg0, %[[POISON0]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:         %[[DESC1:.*]] = llvm.insertvalue %arg1, %[[DESC0]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:         %[[DESC:.*]] = llvm.insertvalue %arg2, %[[DESC1]][2] : !llvm.struct<(ptr, ptr, i64)>
//
// Lower the fir.convert from memref<f32> to !fir.ref<f32> by extracting
// the buffer pointer from the descriptor.
// CHECK:         %[[ALIGNED:.*]] = llvm.extractvalue %[[DESC]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:         %[[OFF:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:         %[[BUF:.*]] = llvm.getelementptr %[[ALIGNED]][%[[OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//
// The second fir.convert (from !fir.ref<f32> back to memref<f32>) lowering
// CHECK:         %[[POISON1:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
// CHECK:         %[[DESC2:.*]] = llvm.insertvalue %[[BUF]], %[[POISON1]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:         %[[DESC3:.*]] = llvm.insertvalue %[[BUF]], %[[DESC2]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:         %[[DESC4:.*]] = llvm.insertvalue %[[ZERO]], %[[DESC3]][2] : !llvm.struct<(ptr, ptr, i64)>
//
// CHECK-NOT:     fir.convert

func.func @memref_to_ref_convert(%arg0: memref<f32>) {
  %0 = fir.convert %arg0 : (memref<f32>) -> !fir.ref<f32>
  %1 = fir.convert %0 : (!fir.ref<f32>) -> memref<f32>
  return
}

// -----

// CHECK-LABEL:   llvm.func @memref_to_memref_convert(
// CHECK-SAME:      %[[ARG0:[^:]*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:[^:]*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG2:[^:]*]]: i64) -> !llvm.struct<(ptr, ptr, i64)> {
// CHECK:           %[[MLIR_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[INSERTVALUE_0:.*]] = llvm.insertvalue %[[ARG0]], %[[MLIR_0]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[INSERTVALUE_1:.*]] = llvm.insertvalue %[[ARG1]], %[[INSERTVALUE_0]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[INSERTVALUE_2:.*]] = llvm.insertvalue %[[ARG2]], %[[INSERTVALUE_1]][2] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[EXTRACTVALUE_0:.*]] = llvm.extractvalue %[[INSERTVALUE_2]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[EXTRACTVALUE_1:.*]] = llvm.extractvalue %[[INSERTVALUE_2]][2] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[EXTRACTVALUE_0]]{{\[}}%[[EXTRACTVALUE_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           %[[MLIR_1:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[INSERTVALUE_3:.*]] = llvm.insertvalue %[[GETELEMENTPTR_0]], %[[MLIR_1]][0] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[INSERTVALUE_4:.*]] = llvm.insertvalue %[[GETELEMENTPTR_0]], %[[INSERTVALUE_3]][1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           %[[MLIR_2:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[INSERTVALUE_5:.*]] = llvm.insertvalue %[[MLIR_2]], %[[INSERTVALUE_4]][2] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:           llvm.return %[[INSERTVALUE_5]] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:         }

func.func @memref_to_memref_convert(%arg0: memref<f32>) -> memref<i1> {
  %0 = fir.convert %arg0 : (memref<f32>) -> memref<i1>
  return %0 : memref<i1>
}

// -----

// CHECK-LABEL:   llvm.func @memref_to_memref_convert(
// CHECK-SAME:      %[[ARG0:[^:]*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG1:[^:]*]]: !llvm.ptr,
// CHECK-SAME:      %[[ARG2:[^:]*]]: i64,
// CHECK-SAME:      %[[ARG3:[^:]*]]: i64,
// CHECK-SAME:      %[[ARG4:[^:]*]]: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
// CHECK:           %[[MLIR_0:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_0:.*]] = llvm.insertvalue %[[ARG0]], %[[MLIR_0]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_1:.*]] = llvm.insertvalue %[[ARG1]], %[[INSERTVALUE_0]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_2:.*]] = llvm.insertvalue %[[ARG2]], %[[INSERTVALUE_1]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_3:.*]] = llvm.insertvalue %[[ARG3]], %[[INSERTVALUE_2]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[INSERTVALUE_4:.*]] = llvm.insertvalue %[[ARG4]], %[[INSERTVALUE_3]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[EXTRACTVALUE_0:.*]] = llvm.extractvalue %[[INSERTVALUE_4]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[EXTRACTVALUE_1:.*]] = llvm.extractvalue %[[INSERTVALUE_4]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[EXTRACTVALUE_0]]{{\[}}%[[EXTRACTVALUE_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           %[[MLIR_1:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[INSERTVALUE_5:.*]] = llvm.insertvalue %[[GETELEMENTPTR_0]], %[[MLIR_1]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[INSERTVALUE_6:.*]] = llvm.insertvalue %[[GETELEMENTPTR_0]], %[[INSERTVALUE_5]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[MLIR_2:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[INSERTVALUE_7:.*]] = llvm.insertvalue %[[MLIR_2]], %[[INSERTVALUE_6]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[MLIR_3:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK:           %[[INSERTVALUE_8:.*]] = llvm.insertvalue %[[MLIR_3]], %[[INSERTVALUE_7]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[MLIR_4:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK:           %[[INSERTVALUE_9:.*]] = llvm.insertvalue %[[MLIR_4]], %[[INSERTVALUE_8]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[MLIR_5:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK:           %[[INSERTVALUE_10:.*]] = llvm.insertvalue %[[MLIR_5]], %[[INSERTVALUE_9]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           %[[MLIR_6:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[INSERTVALUE_11:.*]] = llvm.insertvalue %[[MLIR_6]], %[[INSERTVALUE_10]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:           llvm.return %[[INSERTVALUE_11]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:         }

func.func @memref_to_memref_convert(%arg0: memref<3xf32>) -> memref<2x3xi1> {
  %0 = fir.convert %arg0 : (memref<3xf32>) -> memref<2x3xi1>
  return %0 : memref<2x3xi1>
}
