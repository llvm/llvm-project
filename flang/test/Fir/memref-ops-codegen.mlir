// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu lower-memref=true" --split-input-file %s | FileCheck %s
// RUN: not fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" --split-input-file %s 2>&1 | FileCheck %s --check-prefix=DISABLED

// Test that memref dialect operations, as produced by fir-to-memref and
// lower-affine, are lowered together with the fir.convert ops that bridge
// !fir.ref and memref values when lower-memref is enabled.

// DISABLED: failed to legalize operation 'memref.load'

// CHECK-LABEL: llvm.func @load_scalar(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.ptr) -> i32 {
// CHECK:         %[[ALIGNED:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:         %[[VAL:.*]] = llvm.load %[[ALIGNED]] : !llvm.ptr -> i32
// CHECK:         llvm.return %[[VAL]] : i32
// CHECK-NOT:     memref.
func.func @load_scalar(%arg0: !fir.ref<i32>) -> i32 {
  %0 = fir.convert %arg0 : (!fir.ref<i32>) -> memref<i32>
  %1 = memref.load %0[] : memref<i32>
  return %1 : i32
}

// -----

// CHECK-LABEL: llvm.func @store_scalar(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.ptr, %[[VAL:.*]]: f32) {
// CHECK:         %[[ALIGNED:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr, ptr, i64)>
// CHECK:         llvm.store %[[VAL]], %[[ALIGNED]] : f32, !llvm.ptr
// CHECK-NOT:     memref.
func.func @store_scalar(%arg0: !fir.ref<f32>, %arg1: f32) {
  %0 = fir.convert %arg0 : (!fir.ref<f32>) -> memref<f32>
  memref.store %arg1, %0[] : memref<f32>
  return
}

// -----

// CHECK-LABEL: llvm.func @load_store_static_2d(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.ptr, %[[I:.*]]: i64, %[[J:.*]]: i64) {
// CHECK:         %[[MUL0:.*]] = llvm.mul %[[I]], %{{.*}}
// CHECK:         %[[ADD0:.*]] = llvm.add %[[MUL0]], %[[J]]
// CHECK:         %[[SRC:.*]] = llvm.getelementptr {{.*}}[%[[ADD0]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:         %[[VAL:.*]] = llvm.load %[[SRC]] : !llvm.ptr -> f32
// CHECK:         %[[MUL1:.*]] = llvm.mul %[[J]], %{{.*}}
// CHECK:         %[[ADD1:.*]] = llvm.add %[[MUL1]], %[[I]]
// CHECK:         %[[DST:.*]] = llvm.getelementptr {{.*}}[%[[ADD1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:         llvm.store %[[VAL]], %[[DST]] : f32, !llvm.ptr
// CHECK-NOT:     memref.
func.func @load_store_static_2d(%arg0: !fir.ref<!fir.array<4x8xf32>>, %i: index, %j: index) {
  %0 = fir.convert %arg0 : (!fir.ref<!fir.array<4x8xf32>>) -> memref<4x8xf32>
  %1 = memref.load %0[%i, %j] : memref<4x8xf32>
  memref.store %1, %0[%j, %i] : memref<4x8xf32>
  return
}
