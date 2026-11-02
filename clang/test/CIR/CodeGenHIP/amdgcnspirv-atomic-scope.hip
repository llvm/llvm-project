// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t-cir.ll

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.ll

// RUN: %clang_cc1 -triple spir64-unknown-unknown -x hip -fclangir \
// RUN:            -fcuda-is-device -emit-cir %s -o %t-spir.cir
// RUN: FileCheck --check-prefix=SPIR %s --input-file=%t-spir.cir

#define __device__ __attribute__((device))

// CIR-LABEL: @_Z12singlethreadPff
// CIR: cir.atomic.fetch add relaxed syncscope(hip_single_thread) {{.*}} {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_remote_memory} loc(
// LLVM-LABEL: @_Z12singlethreadPff
// LLVM: atomicrmw fadd {{.*}} syncscope("singlethread") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.no.remote.memory !{{[0-9]+}}{{$}}
// SPIR-LABEL: @_Z12singlethreadPff
// SPIR: cir.atomic.fetch add relaxed syncscope(system) {{.*}} -> !{{cir.float|s32i}} loc(
__device__ float singlethread(float *p, float v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_SINGLETHREAD);
}

// CIR-LABEL: @_Z9wavefrontPii
// CIR: cir.atomic.fetch add relaxed syncscope(hip_wavefront) {{.*}} {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_remote_memory} loc(
// LLVM-LABEL: @_Z9wavefrontPii
// LLVM: atomicrmw add {{.*}} syncscope("subgroup") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.no.remote.memory !{{[0-9]+}}{{$}}
// SPIR-LABEL: @_Z9wavefrontPii
// SPIR: cir.atomic.fetch add relaxed syncscope(system) {{.*}} -> !{{cir.float|s32i}} loc(
__device__ int wavefront(int *p, int v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_WAVEFRONT);
}

// CIR-LABEL: @_Z9workgroupPff
// CIR: cir.atomic.fetch add relaxed syncscope(hip_workgroup) {{.*}} {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_remote_memory} loc(
// LLVM-LABEL: @_Z9workgroupPff
// LLVM: atomicrmw fadd {{.*}} syncscope("workgroup") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.no.remote.memory !{{[0-9]+}}{{$}}
// SPIR-LABEL: @_Z9workgroupPff
// SPIR: cir.atomic.fetch add relaxed syncscope(system) {{.*}} -> !{{cir.float|s32i}} loc(
__device__ float workgroup(float *p, float v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_WORKGROUP);
}

// CIR-LABEL: @_Z5agentPii
// CIR: cir.atomic.fetch add relaxed syncscope(hip_agent) {{.*}} {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_remote_memory} loc(
// LLVM-LABEL: @_Z5agentPii
// LLVM: atomicrmw add {{.*}} syncscope("device") monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.no.remote.memory !{{[0-9]+}}{{$}}
// SPIR-LABEL: @_Z5agentPii
// SPIR: cir.atomic.fetch add relaxed syncscope(system) {{.*}} -> !{{cir.float|s32i}} loc(
__device__ int agent(int *p, int v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT);
}

// CIR-LABEL: @_Z6systemPff
// CIR: cir.atomic.fetch add relaxed syncscope(hip_system) {{.*}} {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_remote_memory} loc(
// LLVM-LABEL: @_Z6systemPff
// LLVM: atomicrmw fadd {{.*}} monotonic, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.no.remote.memory !{{[0-9]+}}{{$}}
// SPIR-LABEL: @_Z6systemPff
// SPIR: cir.atomic.fetch add relaxed syncscope(system) {{.*}} -> !{{cir.float|s32i}} loc(
__device__ float system(float *p, float v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_SYSTEM);
}

// CIR-LABEL: @_Z7cmpxchgPiii
// CIR: cir.atomic.cmpxchg {{.*}} syncscope(hip_agent) {{.*}} -> (!s32i, !cir.bool) loc(
// LLVM-LABEL: @_Z7cmpxchgPiii
// LLVM: cmpxchg {{.*}} syncscope("device") monotonic monotonic, align 4{{$}}
__device__ bool cmpxchg(int *p, int e, int d) {
  return __hip_atomic_compare_exchange_strong(p, &e, d, __ATOMIC_RELAXED,
                                              __ATOMIC_RELAXED,
                                              __HIP_MEMORY_SCOPE_AGENT);
}
