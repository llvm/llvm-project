// REQUIRES: amdgpu-registered-target
// RUN: not %clang_cc1 %s -x hip -fcuda-is-device -o - -emit-llvm -triple=amdgcn-amd-amdhsa 2>&1 | FileCheck %s

void test_host() {
  __UINT32_TYPE__ val32;
  __UINT64_TYPE__ val64;

  // CHECK: error: reference to __device__ function '__builtin_amdgcn_atomic_inc32' in __host__ function
  val32 = __builtin_amdgcn_atomic_inc32(&val32, val32, __ATOMIC_SEQ_CST, "");

  // CHECK: error: reference to __device__ function '__builtin_amdgcn_atomic_inc64' in __host__ function
  val64 = __builtin_amdgcn_atomic_inc64(&val64, val64, __ATOMIC_SEQ_CST, "");

  // CHECK: error: reference to __device__ function '__builtin_amdgcn_atomic_dec32' in __host__ function
  val32 = __builtin_amdgcn_atomic_dec32(&val32, val32, __ATOMIC_SEQ_CST, "");

  // CHECK: error: reference to __device__ function '__builtin_amdgcn_atomic_dec64' in __host__ function
  val64 = __builtin_amdgcn_atomic_dec64(&val64, val64, __ATOMIC_SEQ_CST, "");
}
