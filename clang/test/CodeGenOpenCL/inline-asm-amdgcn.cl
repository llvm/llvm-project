// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -emit-llvm -o - -triple amdgcn %s | FileCheck %s

kernel void test_long(int arg0) {
  long v15_16;
  // CHECK: tail call i64 asm sideeffect "v_lshlrev_b64 v[15:16], 0, $0", "={v[15:16]},v"(i32 %arg0)
  __asm volatile("v_lshlrev_b64 v[15:16], 0, %0" : "={v[15:16]}"(v15_16) : "v"(arg0));
}
