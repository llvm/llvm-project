// RUN: %clang_cc1 -x hip %s -emit-llvm -o - -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx906 -fnative-half-type \
// RUN:   -fnative-half-arguments-and-returns | FileCheck -check-prefixes=CHECK,SAFEIR %s

// RUN: %clang_cc1 -x hip %s -emit-llvm -o - -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx906 -fnative-half-type \
// RUN:   -fnative-half-arguments-and-returns -munsafe-fp-atomics | FileCheck -check-prefixes=CHECK,UNSAFEIR %s

// RUN: %clang_cc1 -x hip %s -O3 -S -o - -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx1100 -fnative-half-type \
// RUN:   -fnative-half-arguments-and-returns | FileCheck -check-prefix=SAFE %s

// RUN: %clang_cc1 -x hip %s -O3 -S -o - -triple=amdgcn-amd-amdhsa \
// RUN:   -fcuda-is-device -target-cpu gfx940 -fnative-half-type \
// RUN:   -fnative-half-arguments-and-returns -munsafe-fp-atomics \
// RUN:   | FileCheck -check-prefix=UNSAFE %s

// REQUIRES: amdgpu-registered-target

#include "Inputs/cuda.h"
#include <stdatomic.h>

__global__ void ffp1(float *p) {
  // CHECK-LABEL: @_Z4ffp1Pf
  // SAFEIR: atomicrmw fadd ptr {{.*}} monotonic, align 4{{$}}
  // SAFEIR: atomicrmw fsub ptr {{.*}} monotonic, align 4{{$}}
  // SAFEIR: atomicrmw fmax ptr {{.*}} monotonic, align 4{{$}}
  // SAFEIR: atomicrmw fmin ptr {{.*}} monotonic, align 4{{$}}
  // SAFEIR: atomicrmw fmax ptr {{.*}} syncscope("agent-one-as") monotonic, align 4{{$}}
  // SAFEIR: atomicrmw fmin ptr {{.*}} syncscope("workgroup-one-as") monotonic, align 4{{$}}

  // UNSAFEIR: atomicrmw fadd ptr {{.*}} monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.ignore.denormal.mode !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fsub ptr {{.*}} monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmax ptr {{.*}} monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmin ptr {{.*}} monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmax ptr {{.*}} syncscope("agent-one-as") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmin ptr {{.*}} syncscope("workgroup-one-as") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}

  // SAFE: _Z4ffp1Pf
  // SAFE: global_atomic_cmpswap
  // SAFE: global_atomic_cmpswap
  // SAFE: global_atomic_cmpswap
  // SAFE: global_atomic_cmpswap
  // SAFE: global_atomic_cmpswap
  // SAFE: global_atomic_cmpswap

  // UNSAFE: _Z4ffp1Pf
  // UNSAFE: global_atomic_add_f32
  // UNSAFE: global_atomic_cmpswap
  // UNSAFE: global_atomic_cmpswap
  // UNSAFE: global_atomic_cmpswap
  // UNSAFE: global_atomic_cmpswap
  // UNSAFE: global_atomic_cmpswap

  __atomic_fetch_add(p, 1.0f, memory_order_relaxed);
  __atomic_fetch_sub(p, 1.0f, memory_order_relaxed);
  __atomic_fetch_max(p, 1.0f, memory_order_relaxed);
  __atomic_fetch_min(p, 1.0f, memory_order_relaxed);
  __hip_atomic_fetch_max(p, 1.0f, memory_order_relaxed, __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_fetch_min(p, 1.0f, memory_order_relaxed, __HIP_MEMORY_SCOPE_WORKGROUP);
}

__global__ void ffp2(double *p) {
  // CHECK-LABEL: @_Z4ffp2Pd
  // SAFEIR: atomicrmw fadd ptr {{.*}} monotonic, align 8{{$}}
  // SAFEIR: atomicrmw fsub ptr {{.*}} monotonic, align 8{{$}}
  // SAFEIR: atomicrmw fmax ptr {{.*}} monotonic, align 8{{$}}
  // SAFEIR: atomicrmw fmin ptr {{.*}} monotonic, align 8{{$}}
  // SAFEIR: atomicrmw fmax ptr {{.*}} syncscope("agent-one-as") monotonic, align 8{{$}}
  // SAFEIR: atomicrmw fmin ptr {{.*}} syncscope("workgroup-one-as") monotonic, align 8{{$}}

  // UNSAFEIR: atomicrmw fadd ptr {{.*}} monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fsub ptr {{.*}} monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmax ptr {{.*}} monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmin ptr {{.*}} monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmax ptr {{.*}} syncscope("agent-one-as") monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmin ptr {{.*}} syncscope("workgroup-one-as") monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}

  // SAFE-LABEL: @_Z4ffp2Pd
  // SAFE: global_atomic_cmpswap_b64
  // SAFE: global_atomic_cmpswap_b64
  // SAFE: global_atomic_cmpswap_b64
  // SAFE: global_atomic_cmpswap_b64
  // SAFE: global_atomic_cmpswap_b64
  // SAFE: global_atomic_cmpswap_b64

  // UNSAFE-LABEL: @_Z4ffp2Pd
  // UNSAFE: global_atomic_add_f64
  // UNSAFE: global_atomic_cmpswap_x2
  // UNSAFE: global_atomic_max_f64
  // UNSAFE: global_atomic_min_f64
  // UNSAFE: global_atomic_max_f64
  // UNSAFE: global_atomic_min_f64
  __atomic_fetch_add(p, 1.0, memory_order_relaxed);
  __atomic_fetch_sub(p, 1.0, memory_order_relaxed);
  __atomic_fetch_max(p, 1.0, memory_order_relaxed);
  __atomic_fetch_min(p, 1.0, memory_order_relaxed);
  __hip_atomic_fetch_max(p, 1.0f, memory_order_relaxed, __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_fetch_min(p, 1.0f, memory_order_relaxed, __HIP_MEMORY_SCOPE_WORKGROUP);
}

// long double is the same as double for amdgcn.
__global__ void ffp3(long double *p) {
  // CHECK-LABEL: @_Z4ffp3Pe
  // SAFEIR: atomicrmw fadd ptr {{.*}} monotonic, align 8{{$}}
  // SAFEIR: atomicrmw fsub ptr {{.*}} monotonic, align 8{{$}}
  // SAFEIR: atomicrmw fmax ptr {{.*}} monotonic, align 8{{$}}
  // SAFEIR: atomicrmw fmin ptr {{.*}} monotonic, align 8{{$}}
  // SAFEIR: atomicrmw fmax ptr {{.*}} syncscope("agent-one-as") monotonic, align 8{{$}}
  // SAFEIR: atomicrmw fmin ptr {{.*}} syncscope("workgroup-one-as") monotonic, align 8{{$}}

  // UNSAFEIR: atomicrmw fadd ptr {{.*}} monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fsub ptr {{.*}} monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmax ptr {{.*}} monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmin ptr {{.*}} monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmax ptr {{.*}} syncscope("agent-one-as") monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmin ptr {{.*}} syncscope("workgroup-one-as") monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}

  // SAFE-LABEL: @_Z4ffp3Pe
  // SAFE: global_atomic_cmpswap_b64
  // SAFE: global_atomic_cmpswap_b64
  // SAFE: global_atomic_cmpswap_b64
  // SAFE: global_atomic_cmpswap_b64
  // SAFE: global_atomic_cmpswap_b64
  // UNSAFE-LABEL: @_Z4ffp3Pe
  // UNSAFE: global_atomic_cmpswap_x2
  // UNSAFE: global_atomic_max_f64
  // UNSAFE: global_atomic_min_f64
  // UNSAFE: global_atomic_max_f64
  // UNSAFE: global_atomic_min_f64
  __atomic_fetch_add(p, 1.0L, memory_order_relaxed);
  __atomic_fetch_sub(p, 1.0L, memory_order_relaxed);
  __atomic_fetch_max(p, 1.0L, memory_order_relaxed);
  __atomic_fetch_min(p, 1.0L, memory_order_relaxed);
  __hip_atomic_fetch_max(p, 1.0f, memory_order_relaxed, __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_fetch_min(p, 1.0f, memory_order_relaxed, __HIP_MEMORY_SCOPE_WORKGROUP);
}

__device__ double ffp4(double *p, float f) {
  // CHECK-LABEL: @_Z4ffp4Pdf
  // CHECK: fpext float {{.*}} to double
  // SAFEIR: atomicrmw fsub ptr {{.*}} monotonic, align 8{{$}}
  // UNSAFEIR: atomicrmw fsub ptr {{.*}} monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  return __atomic_fetch_sub(p, f, memory_order_relaxed);
}

__device__ double ffp5(double *p, int i) {
  // CHECK-LABEL: @_Z4ffp5Pdi
  // CHECK: sitofp i32 {{.*}} to double
  // SAFEIR: atomicrmw fsub ptr {{.*}} monotonic, align 8{{$}}
  // UNSAFEIR: atomicrmw fsub ptr {{.*}} monotonic, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  return __atomic_fetch_sub(p, i, memory_order_relaxed);
}

__global__ void ffp6(_Float16 *p) {
  // CHECK-LABEL: @_Z4ffp6PDF16
  // SAFEIR: atomicrmw fadd ptr {{.*}} monotonic, align 2{{$}}
  // SAFEIR: atomicrmw fsub ptr {{.*}} monotonic, align 2{{$}}
  // SAFEIR: atomicrmw fmax ptr {{.*}} monotonic, align 2{{$}}
  // SAFEIR: atomicrmw fmin ptr {{.*}} monotonic, align 2{{$}}
  // SAFEIR: atomicrmw fmax ptr {{.*}} syncscope("agent-one-as") monotonic, align 2{{$}}
  // SAFEIR: atomicrmw fmin ptr {{.*}} syncscope("workgroup-one-as") monotonic, align 2{{$}}

  // UNSAFEIR: atomicrmw fadd ptr {{.*}} monotonic, align 2, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fsub ptr {{.*}} monotonic, align 2, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmax ptr {{.*}} monotonic, align 2, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmin ptr {{.*}} monotonic, align 2, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmax ptr {{.*}} syncscope("agent-one-as") monotonic, align 2, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  // UNSAFEIR: atomicrmw fmin ptr {{.*}} syncscope("workgroup-one-as") monotonic, align 2, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}

  // SAFE: _Z4ffp6PDF16
  // SAFE: global_atomic_cmpswap
  // SAFE: global_atomic_cmpswap
  // SAFE: global_atomic_cmpswap
  // SAFE: global_atomic_cmpswap
  // SAFE: global_atomic_cmpswap
  // SAFE: global_atomic_cmpswap

  // UNSAFE: _Z4ffp6PDF16
  // UNSAFE: global_atomic_cmpswap
  // UNSAFE: global_atomic_cmpswap
  // UNSAFE: global_atomic_cmpswap
  // UNSAFE: global_atomic_cmpswap
  // UNSAFE: global_atomic_cmpswap
  // UNSAFE: global_atomic_cmpswap
  __atomic_fetch_add(p, 1.0, memory_order_relaxed);
  __atomic_fetch_sub(p, 1.0, memory_order_relaxed);
  __atomic_fetch_max(p, 1.0, memory_order_relaxed);
  __atomic_fetch_min(p, 1.0, memory_order_relaxed);
  __hip_atomic_fetch_max(p, 1.0f, memory_order_relaxed, __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_fetch_min(p, 1.0f, memory_order_relaxed, __HIP_MEMORY_SCOPE_WORKGROUP);
}
