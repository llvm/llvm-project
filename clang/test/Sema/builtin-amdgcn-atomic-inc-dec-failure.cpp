// REQUIRES: amdgpu-registered-target
// RUN: not %clang_cc1 %s -x hip -fcuda-is-device -o - -emit-llvm -triple=amdgcn-amd-amdhsa 2>&1 | FileCheck %s

void test_host() {
  int val;

  // CHECK: error: reference to __device__ function '__builtin_amdgcn_atomic_inc32' in __host__ function
  val = __builtin_amdgcn_atomic_inc32(&val, val, __ATOMIC_SEQ_CST, "");

  // CHECK: error: reference to __device__ function '__builtin_amdgcn_atomic_inc64' in __host__ function
  val = __builtin_amdgcn_atomic_inc64(&val, val, __ATOMIC_SEQ_CST, "");

  // CHECK: error: reference to __device__ function '__builtin_amdgcn_atomic_dec32' in __host__ function
  val = __builtin_amdgcn_atomic_dec32(&val, val, __ATOMIC_SEQ_CST, "");

  // CHECK: error: reference to __device__ function '__builtin_amdgcn_atomic_dec64' in __host__ function
  val = __builtin_amdgcn_atomic_dec64(&val, val, __ATOMIC_SEQ_CST, "");
}
