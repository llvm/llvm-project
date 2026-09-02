// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// CHECK: target triple
// Note: the original bug was just that the test asserted
// because it attempted to emit an empty statement, so check-lines
// are effectively irrelevant, but included for completeness.

void foo() {
#pragma acc parallel
  _Alignas(4);
  // CHECK-NOT: acc parallel
#pragma acc loop
  _Alignas(4);
  // CHECK-NOT: acc loop
#pragma acc kernels loop
  _Alignas(4);
  // CHECK-NOT: acc kernels
#pragma acc data
  _Alignas(4);
  // CHECK-NOT: acc data
#pragma acc host_data
  _Alignas(4);
  // CHECK-NOT: acc host
#pragma acc atomic
  _Alignas(4);
  // CHECK-NOT: acc atomic
}

