// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

int a[101], b[101];

void constantLoopBound() {
  for (int i = 0; i < 100; ++i)
    a[i] = 3;
}
// CHECK-LABEL: func.func @_Z17constantLoopBoundv() {
// CHECK: memref.alloca_scope  {
// CHECK-NOT: {{.*}} = memref.alloca() {alignment = 4 : i64} : memref<1xi32>
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NOT: memref.store %[[C0]], {{.*}}[{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK: %[[C100:.*]] = arith.constant 100 : i32
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C100]] step %[[C1]] : i32 {
// CHECK:   %[[C3:.*]] = arith.constant 3 : i32
// CHECK:   %[[BASE:.*]] = memref.get_global @a : memref<101xi32>
// CHECK:   %[[INDEX:.*]] = arith.index_cast %[[I]] : i32 to index
// CHECK:   memref.store %[[C3]], %[[BASE]][%[[INDEX]]] : memref<101xi32>
// CHECK: }
// CHECK: }

void constantLoopBound_LE() {
  for (int i = 0; i <= 100; ++i)
    a[i] = 3;
}
// CHECK-LABEL: func.func @_Z20constantLoopBound_LEv() {
// CHECK: memref.alloca_scope  {
// CHECK-NOT: {{.*}} = memref.alloca() {alignment = 4 : i64} : memref<1xi32>
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NOT: memref.store %[[C0]], {{.*}}[{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK: %[[C100:.*]] = arith.constant 100 : i32
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[C101:.*]] = arith.addi %c100_i32, %c1_i32 : i32
// CHECK: %[[C1_STEP:.*]] = arith.constant 1 : i32
// CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C101]] step %[[C1_STEP]] : i32 {
// CHECK:   %[[C3:.*]] = arith.constant 3 : i32
// CHECK:   %[[BASE:.*]] = memref.get_global @a : memref<101xi32>
// CHECK:   %[[INDEX:.*]] = arith.index_cast %[[I]] : i32 to index
// CHECK:   memref.store %[[C3]], %[[BASE]][%[[INDEX]]] : memref<101xi32>
// CHECK: }
// CHECK: }

void variableLoopBound(int l, int u) {
  for (int i = l; i < u; ++i)
    a[i] = 3;
}
// CHECK-LABEL: func.func @_Z17variableLoopBoundii
// CHECK: memref.store %arg0, %alloca[{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK: memref.store %arg1, %alloca_0[{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK: memref.alloca_scope  {
// CHECK-NOT: {{.*}} = memref.alloca() {alignment = 4 : i64} : memref<1xi32>
// CHECK: %[[LOWER:.*]] = memref.load %alloca[{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK-NOT: memref.store %[[LOWER]], {{.*}}[{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK: %[[UPPER:.*]] = memref.load %alloca_0[{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: scf.for %[[I:.*]] = %[[LOWER]] to %[[UPPER]] step %[[C1]] : i32 {
// CHECK:   %[[C3:.*]] = arith.constant 3 : i32
// CHECK:   %[[BASE:.*]] = memref.get_global @a : memref<101xi32>
// CHECK:   %[[INDEX:.*]] = arith.index_cast %[[I]] : i32 to index
// CHECK:   memref.store %[[C3]], %[[BASE]][%[[INDEX]]] : memref<101xi32>
// CHECK: }
// CHECK: }

void variableLoopBound_LE(int l, int u) {
  for (int i = l; i <= u; i+=4)
    a[i] = 3;
}
// CHECK-LABEL: func.func @_Z20variableLoopBound_LEii
// CHECK: memref.store %arg0, %alloca[{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK: memref.store %arg1, %alloca_0[{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK: memref.alloca_scope  {
// CHECK-NOT: {{.*}} = memref.alloca() {alignment = 4 : i64} : memref<1xi32>
// CHECK: %[[LOWER:.*]] = memref.load %alloca[{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK-NOT: memref.store %[[LOWER]], {{.*}}[{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK: %[[UPPER_DEC_1:.*]] = memref.load %alloca_0[{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[UPPER:.*]] = arith.addi %[[UPPER_DEC_1]], %[[C1]] : i32
// CHECK: %[[C4:.*]] = arith.constant 4 : i32
// CHECK: scf.for %[[I:.*]] = %[[LOWER]] to %[[UPPER]] step %[[C4]] : i32 {
// CHECK:   %[[C3:.*]] = arith.constant 3 : i32
// CHECK:   %[[BASE:.*]] = memref.get_global @a : memref<101xi32>
// CHECK:   %[[INDEX:.*]] = arith.index_cast %[[I]] : i32 to index
// CHECK:   memref.store %[[C3]], %[[BASE]][%[[INDEX]]] : memref<101xi32>
// CHECK: }
// CHECK: }

void incArray() {
  for (int i = 0; i < 100; ++i)
    a[i] += b[i];
}
// CHECK-LABEL: func.func @_Z8incArrayv() {
// CHECK: memref.alloca_scope  {
// CHECK-NOT: {{.*}} = memref.alloca() {alignment = 4 : i64} : memref<1xi32>
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-NOT: memref.store %[[C0]], {{.*}}[{{%c0(_[0-9]+)?}}] : memref<1xi32>
// CHECK: %[[C100:.*]] = arith.constant 100 : i32
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C100]] step %[[C1]] : i32 {
// CHECK:   %[[B:.*]] = memref.get_global @b : memref<101xi32>
// CHECK:   %[[INDEX_2:.*]] = arith.index_cast %[[I]] : i32 to index
// CHECK:   %[[B_VALUE:.*]] = memref.load %[[B]][%[[INDEX_2]]] : memref<101xi32>
// CHECK:   %[[A:.*]] = memref.get_global @a : memref<101xi32>
// CHECK:   %[[INDEX_1:.*]] = arith.index_cast %[[I]] : i32 to index
// CHECK:   %[[A_VALUE:.*]] = memref.load %[[A]][%[[INDEX_1]]] : memref<101xi32>
// CHECK:   %[[SUM:.*]] = arith.addi %[[A_VALUE]], %[[B_VALUE]] : i32
// CHECK:   memref.store %[[SUM]], %[[A]][%[[INDEX_1]]] : memref<101xi32>
// CHECK: }
// CHECK: }
