// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s -check-prefix=MLIR

void foo() {
  char s1[] = "Hello";
}

// MLIR:     memref.global "private" constant @[[GLOBAL_ARRAY:.*]] : memref<6xi8> = dense<[72, 101, 108, 108, 111, 0]>
// MLIR:     @_Z3foov() {
// MLIR-DAG: %[[ALLOCA:.*]] = memref.alloca() {alignment = 1 : i64} : memref<6xi8>
// MLIR-DAG: %[[SOURCE:.*]] = memref.get_global @[[GLOBAL_ARRAY]] : memref<6xi8>
// MLIR:     memref.copy %[[SOURCE]], %[[ALLOCA]] : memref<6xi8> to memref<6xi8>
