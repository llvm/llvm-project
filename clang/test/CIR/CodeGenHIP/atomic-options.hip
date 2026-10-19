// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu9.0a-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-cir %s -o - \
// RUN: | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple amdgpu9.0a-amd-amdhsa -x hip -fclangir -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple amdgpu9.0a-amd-amdhsa -x hip -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

// CIR-LABEL: @_Z9plain_addPff
// CIR: cir.atomic.fetch add relaxed syncscope(hip_agent) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_private_addrspace, cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z9plain_addPff
// LLVM: atomicrmw fadd ptr %{{.+}}, float %{{.+}} syncscope("agent") monotonic, align 4{{.*}}!amdgpu.{{(no.remote.memory|no.fine.grained.memory)}} !{{[0-9]+}}{{.*}}!amdgpu.{{(no.remote.memory|no.fine.grained.memory)}}
__device__ float plain_add(float *p, float v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, 4);
}

// CIR-LABEL: @_Z10unsafe_addPff
// CIR: cir.atomic.fetch add relaxed syncscope(hip_agent) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float {cir.amdgpu_no_private_addrspace, cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z10unsafe_addPff
// LLVM: atomicrmw fadd ptr %{{.+}}, float %{{.+}} syncscope("agent") monotonic
// LLVM-NOT: amdgpu.no.fine.grained.memory
__device__ float unsafe_add(float *p, float v) {
  [[clang::atomic(fine_grained_memory)]] {
    return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, 4);
  }
}

// CIR-LABEL: @_Z9no_fg_addPff
// CIR: cir.atomic.fetch add relaxed syncscope(hip_agent) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float {cir.amdgpu_ignore_denormal_mode, cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_private_addrspace, cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z9no_fg_addPff
// LLVM: atomicrmw fadd ptr %{{.+}}, float %{{.+}} syncscope("agent") monotonic, align 4{{.*}}!atomic.ignore.denormal.mode !{{[0-9]+}}, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.no.remote.memory
__device__ float no_fg_add(float *p, float v) {
  [[clang::atomic(no_fine_grained_memory, ignore_denormal_mode)]] {
    return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, 4);
  }
}

// CIR-LABEL: @_Z9no_fg_addPdd
// CIR: cir.atomic.fetch add relaxed syncscope(hip_agent) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.double>, !cir.double) -> !cir.double {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_private_addrspace, cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z9no_fg_addPdd
// LLVM: atomicrmw fadd ptr %{{.+}}, double %{{.+}} syncscope("agent") monotonic
// LLVM-NOT: atomic.ignore.denormal.mode
__device__ double no_fg_add(double *p, double v) {
  [[clang::atomic(no_fine_grained_memory, ignore_denormal_mode)]] {
    return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, 4);
  }
}

// CIR-LABEL: @_Z12workgroup_ddPdd
// CIR: cir.atomic.fetch add relaxed syncscope(hip_workgroup) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.double>, !cir.double) -> !cir.double {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_private_addrspace, cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z12workgroup_ddPdd
// LLVM: atomicrmw fadd ptr %{{.+}}, double %{{.+}} syncscope("workgroup") monotonic
__device__ double workgroup_dd(double *p, double v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, 3);
}

// CIR-LABEL: @_Z12after_scopedPff
// CIR: cir.atomic.fetch add relaxed syncscope(hip_agent) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_private_addrspace, cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z12after_scopedPff
// LLVM: atomicrmw fadd ptr %{{.+}}, float %{{.+}} syncscope("agent") monotonic, align 4{{.*}}!amdgpu.no.fine.grained.memory
__device__ float after_scoped(float *p, float v) {
  [[clang::atomic(fine_grained_memory)]] { (void)0; }
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, 4);
}

// CIR-LABEL: @_Z4xchgPii
// CIR: cir.atomic.xchg relaxed syncscope(hip_agent) %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i {cir.amdgpu_no_private_addrspace}
// LLVM-LABEL: @_Z4xchgPii
// LLVM: atomicrmw xchg ptr %{{.+}}, i32 %{{.+}} syncscope("agent") monotonic, align 4, !noalias.addrspace ![[#NOPRIV:]]
__device__ int xchg(int *p, int v) {
  return __hip_atomic_exchange(p, v, __ATOMIC_RELAXED, 4);
}

// CIR-LABEL: @_Z7cmpxchgPiii
// CIR: cir.atomic.cmpxchg success(relaxed) failure(relaxed) syncscope(hip_agent) %{{.+}}, %{{.+}}, %{{.+}} align(4) : (!cir.ptr<!s32i>, !s32i, !s32i) -> (!s32i, !cir.bool) {cir.amdgpu_no_private_addrspace}
// LLVM-LABEL: @_Z7cmpxchgPiii
// LLVM: cmpxchg ptr %{{.+}}, i32 %{{.+}}, i32 %{{.+}} syncscope("agent") monotonic monotonic, align 4, !noalias.addrspace ![[#NOPRIV]]{{$}}
__device__ bool cmpxchg(int *p, int e, int d) {
  return __hip_atomic_compare_exchange_strong(p, &e, d, __ATOMIC_RELAXED,
                                              __ATOMIC_RELAXED, 4);
}

// CIR-LABEL: @_Z10global_addPU3AS1ii
// CIR: cir.atomic.fetch add relaxed syncscope(hip_agent) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i, target_address_space(1)>, !s32i) -> !s32i {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_private_addrspace, cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z10global_addPU3AS1ii
// LLVM: atomicrmw add ptr addrspace(1) %{{.+}}, i32 %{{.+}} syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory
__device__ int global_add(__attribute__((address_space(1))) int *p, int v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, 4);
}

// CIR-LABEL: @_Z7gnu_addPii
// CIR: cir.atomic.fetch add relaxed syncscope(system) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!s32i>, !s32i) -> !s32i {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z7gnu_addPii
// LLVM: atomicrmw add ptr %{{.+}}, i32 %{{.+}} monotonic, align 4, !amdgpu.no.fine.grained.memory
__device__ int gnu_add(int *p, int v) {
  return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
}

// LLVM: ![[#NOPRIV]] = !{i32 5, i32 6}
