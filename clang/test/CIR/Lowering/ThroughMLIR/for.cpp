// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

int a[101], b[101];

void constantLoopBound() {
  for (int i = 0; i < 100; ++i)
    a[i] = 3;
}
// CHECK-LABEL: func.func @_Z17constantLoopBoundv() {
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[C100:.*]] = arith.constant 100 : i32
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C100]] step %[[C1]] : i32 {
// CHECK:   %[[C3:.*]] = arith.constant 3 : i32
// CHECK:   %[[BASE:.*]] = memref.get_global @a : memref<101xi32>
// CHECK:   %[[C0_i32:.*]] = arith.constant 0 : i32
// CHECK:   %[[IV:.*]] = arith.addi %[[I]], %[[C0_i32]] : i32
// CHECK:   %[[INDEX:.*]] = arith.index_cast %[[IV]] : i32 to index
// CHECK:   memref.store %[[C3]], %[[BASE]][%[[INDEX]]] : memref<101xi32>
// CHECK: }

void constantLoopBound_LE() {
  for (int i = 0; i <= 100; ++i)
    a[i] = 3;
}
// CHECK-LABEL: func.func @_Z20constantLoopBound_LEv() {
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[C100:.*]] = arith.constant 100 : i32
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[C101:.*]] = arith.addi %c100_i32, %c1_i32 : i32
// CHECK: %[[C1_STEP:.*]] = arith.constant 1 : i32
// CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C101]] step %[[C1_STEP]] : i32 {
// CHECK:   %[[C3:.*]] = arith.constant 3 : i32
// CHECK:   %[[BASE:.*]] = memref.get_global @a : memref<101xi32>
// CHECK:   %[[C0_i32:.*]] = arith.constant 0 : i32
// CHECK:   %[[IV:.*]] = arith.addi %[[I]], %[[C0_i32]] : i32
// CHECK:   %[[INDEX:.*]] = arith.index_cast %[[IV]] : i32 to index
// CHECK:   memref.store %[[C3]], %[[BASE]][%[[INDEX]]] : memref<101xi32>
// CHECK: }

void variableLoopBound(int l, int u) {
  for (int i = l; i < u; ++i)
    a[i] = 3;
}
// CHECK-LABEL: func.func @_Z17variableLoopBoundii
// CHECK: memref.store %arg0, %alloca[] : memref<i32>
// CHECK: memref.store %arg1, %alloca_0[] : memref<i32>
// CHECK: %[[LOWER:.*]] = memref.load %alloca[] : memref<i32>
// CHECK: %[[UPPER:.*]] = memref.load %alloca_0[] : memref<i32>
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: scf.for %[[I:.*]] = %[[LOWER]] to %[[UPPER]] step %[[C1]] : i32 {
// CHECK:   %[[C3:.*]] = arith.constant 3 : i32
// CHECK:   %[[BASE:.*]] = memref.get_global @a : memref<101xi32>
// CHECK:   %[[C0:.*]] = arith.constant 0 : i32
// CHECK:   %[[IV:.*]] = arith.addi %[[I]], %[[C0]] : i32
// CHECK:   %[[INDEX:.*]] = arith.index_cast %[[IV]] : i32 to index
// CHECK:   memref.store %[[C3]], %[[BASE]][%[[INDEX]]] : memref<101xi32>
// CHECK: }

void ariableLoopBound_LE(int l, int u) {
  for (int i = l; i <= u; i+=4)
    a[i] = 3;
}
// CHECK-LABEL: func.func @_Z19ariableLoopBound_LEii
// CHECK: memref.store %arg0, %alloca[] : memref<i32>
// CHECK: memref.store %arg1, %alloca_0[] : memref<i32>
// CHECK: %[[LOWER:.*]] = memref.load %alloca[] : memref<i32>
// CHECK: %[[UPPER_DEC_1:.*]] = memref.load %alloca_0[] : memref<i32>
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[UPPER:.*]] = arith.addi %[[UPPER_DEC_1]], %[[C1]] : i32
// CHECK: %[[C4:.*]] = arith.constant 4 : i32
// CHECK: scf.for %[[I:.*]] = %[[LOWER]] to %[[UPPER]] step %[[C4]] : i32 {
// CHECK:   %[[C3:.*]] = arith.constant 3 : i32
// CHECK:   %[[BASE:.*]] = memref.get_global @a : memref<101xi32>
// CHECK:   %[[C0:.*]] = arith.constant 0 : i32
// CHECK:   %[[IV:.*]] = arith.addi %[[I]], %[[C0]] : i32
// CHECK:   %[[INDEX:.*]] = arith.index_cast %[[IV]] : i32 to index
// CHECK:   memref.store %[[C3]], %[[BASE]][%[[INDEX]]] : memref<101xi32>
// CHECK: }

void incArray() {
  for (int i = 0; i < 100; ++i)
    a[i] += b[i];
}
// CHECK-LABEL: func.func @_Z8incArrayv() {
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[C100:.*]] = arith.constant 100 : i32
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C100]] step %[[C1]] : i32 {
// CHECK:   %[[B:.*]] = memref.get_global @b : memref<101xi32>
// CHECK:   %[[C0_2:.*]] = arith.constant 0 : i32
// CHECK:   %[[IV2:.*]] = arith.addi %[[I]], %[[C0_2]] : i32
// CHECK:   %[[INDEX_2:.*]] = arith.index_cast %[[IV2]] : i32 to index
// CHECK:   %[[B_VALUE:.*]] = memref.load %[[B]][%[[INDEX_2]]] : memref<101xi32>
// CHECK:   %[[A:.*]] = memref.get_global @a : memref<101xi32>
// CHECK:   %[[C0_1:.*]] = arith.constant 0 : i32
// CHECK:   %[[IV1:.*]] = arith.addi %[[I]], %[[C0_1]] : i32
// CHECK:   %[[INDEX_1:.*]] = arith.index_cast %[[IV1]] : i32 to index
// CHECK:   %[[A_VALUE:.*]] = memref.load %[[A]][%[[INDEX_1]]] : memref<101xi32>
// CHECK:   %[[SUM:.*]] = arith.addi %[[A_VALUE]], %[[B_VALUE]] : i32
// CHECK:   memref.store %[[SUM]], %[[A]][%[[INDEX_1]]] : memref<101xi32>
// CHECK: }
