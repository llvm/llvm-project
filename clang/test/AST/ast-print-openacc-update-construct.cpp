// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s
void uses() {
  // CHECK: #pragma acc update
#pragma acc update
}
