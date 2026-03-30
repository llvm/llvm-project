// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +sse2 -O0 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +f16c -O0 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK

_Float16 test_fneg(_Float16 x) {
  // CHECK-LABEL: define {{.*}} @test_fneg
  // CHECK-NOT: fpext
  // CHECK: fneg half
  // CHECK-NOT: fptrunc
  return -x;
}

_Float16 test_fabs(_Float16 x) {
  // CHECK-LABEL: define {{.*}} @test_fabs
  // CHECK-NOT: fpext
  // CHECK: call half @llvm.fabs.f16(half
  // CHECK-NOT: fptrunc
  return __builtin_fabsf16(x);
}
