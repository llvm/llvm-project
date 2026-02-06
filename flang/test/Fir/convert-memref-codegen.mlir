// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s -o - | FileCheck %s

// This test ensures that the FIR CodeGen ConvertOpConversion
// properly lowers fir.convert when either the source or the destination
// type is a memref.

module {
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
}


