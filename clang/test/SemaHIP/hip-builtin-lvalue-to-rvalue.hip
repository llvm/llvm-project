// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -std=c++20 -triple amdgcn -target-cpu gfx1200 -fsyntax-only -fcuda-is-device -ast-dump %s 2>&1 | FileCheck %s

// expected-no-diagnostics

// Test that HIP builtins correctly perform lvalue-to-rvalue conversion on pointer arg.

#define __device__ __attribute__((device))

typedef __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16 fp16x8_t;

// CHECK: ImplicitCastExpr {{.*}} <AddressSpaceConversion>
// CHECK-NEXT: ImplicitCastExpr {{.*}} <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue Var {{.*}} 'glb_ptr'
__device__ fp16x8_t test_global_load_transpose(const _Float16 *in_ptr) {
  fp16x8_t *glb_ptr = reinterpret_cast<fp16x8_t *>(
      reinterpret_cast<__UINTPTR_TYPE__>(in_ptr));
  return __builtin_amdgcn_global_load_tr_b128_v8f16(glb_ptr);
}
