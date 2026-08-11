// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// CHECK: target triple

void foo() {
#pragma acc parallel
  _Alignas(4);
  // CHECK-NOT: parallel
#pragma acc loop
  _Alignas(4);
  // CHECK-NOT: loop
#pragma acc kernels loop
  _Alignas(4);
  // CHECK-NOT: kernels
#pragma acc data
  _Alignas(4);
  // CHECK-NOT: data 
#pragma acc host_data
  _Alignas(4);
  // CHECK-NOT: host 
#pragma acc atomic
  _Alignas(4);
  // CHECK-NOT: atomic 
}

