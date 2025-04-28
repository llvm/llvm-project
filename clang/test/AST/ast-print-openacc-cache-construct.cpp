// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s

void foo() {
  int Array[5];
  // CHECK: #pragma acc cache(readonly: Array[1], Array[1:2])
  #pragma acc cache(readonly:Array[1], Array[1:2])
  // CHECK: #pragma acc cache(Array[1], Array[1:2])
  #pragma acc cache(Array[1], Array[1:2])
}
