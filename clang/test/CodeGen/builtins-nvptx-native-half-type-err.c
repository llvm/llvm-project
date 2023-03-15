// REQUIRES: nvptx-registered-target
//
// RUN: not %clang_cc1 -fsyntax-only -ffp-contract=off -triple nvptx-unknown-unknown -target-cpu \
// RUN:   sm_75 -target-feature +ptx70 -fcuda-is-device -x cuda -emit-llvm -o - %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ERROR %s

#define __device__ __attribute__((device))
typedef __fp16 __fp16v2 __attribute__((ext_vector_type(2)));

__device__ void nvvm_ldg_ldu_native_half_types(const void *p) {
  __nvvm_ldg_h((const __fp16 *)p);
  __nvvm_ldg_h2((const __fp16v2 *)p);

  __nvvm_ldu_h((const __fp16 *)p);
  __nvvm_ldu_h2((const __fp16v2 *)p);
}

// CHECK-ERROR: error: __nvvm_ldg_h requires native half type support.
// CHECK-ERROR: error: __nvvm_ldg_h2 requires native half type support.
// CHECK-ERROR: error: __nvvm_ldu_h requires native half type support.
// CHECK-ERROR: error: __nvvm_ldu_h2 requires native half type support.
