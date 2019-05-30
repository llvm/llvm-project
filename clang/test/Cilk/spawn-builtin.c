// Check that builins can be spawned.  Thanks to Brian Wheatman for originally finding this bug.
//
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -fcilkplus -ftapir=none -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo(float *A, float *B, int n) {
  _Cilk_spawn __builtin_memcpy(A, B, sizeof(float) * n/2);
  __builtin_memcpy(A+n/2, B+n/2, sizeof(float) * (n-n/2));
  _Cilk_sync;
}

// CHECK: detach within %[[SYNCREG:.+]], label %[[DETBLOCK:.+]], label %[[CONT:.+]]
// CHECK: [[DETBLOCK]]:
// CHECK-NEXT: call void @llvm.memcpy
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONT]]
// CHECK: call void @llvm.memcpy
// CHECK: sync within %[[SYNCREG]]
