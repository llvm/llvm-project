// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s
void uses(bool cond) {
  // CHECK: #pragma acc update
#pragma acc update

// CHECK: #pragma acc update if_present
#pragma acc update if_present
// CHECK: #pragma acc update if(cond)
#pragma acc update if(cond)
}
