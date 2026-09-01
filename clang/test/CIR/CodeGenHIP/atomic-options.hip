// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx90a -x hip \
// RUN: -std=c++11 -fclangir -fcuda-is-device -emit-cir %s -o - \
// RUN: | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx90a -x hip \
// RUN: -std=c++11 -fclangir -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx90a -x hip \
// RUN: -std=c++11 -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck --check-prefix=LLVM %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

// CIR-LABEL: @_Z9plain_addPff
// CIR: cir.atomic.fetch add relaxed syncscope(hip_agent) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z9plain_addPff
// LLVM: atomicrmw fadd ptr %{{.+}}, float %{{.+}} syncscope("agent") monotonic, align 4{{.*}}!amdgpu.{{(no.remote.memory|no.fine.grained.memory)}} !{{[0-9]+}}{{.*}}!amdgpu.{{(no.remote.memory|no.fine.grained.memory)}}
__device__ float plain_add(float *p, float v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, 4);
}

// CIR-LABEL: @_Z10unsafe_addPff
// CIR: cir.atomic.fetch add relaxed syncscope(hip_agent) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float {cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z10unsafe_addPff
// LLVM: atomicrmw fadd ptr %{{.+}}, float %{{.+}} syncscope("agent") monotonic
// LLVM-NOT: amdgpu.no.fine.grained.memory
__device__ float unsafe_add(float *p, float v) {
  [[clang::atomic(fine_grained_memory)]] {
    return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, 4);
  }
}

// CIR-LABEL: @_Z9no_fg_addPff
// CIR: cir.atomic.fetch add relaxed syncscope(hip_agent) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float {cir.amdgpu_ignore_denormal_mode, cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z9no_fg_addPff
// LLVM: atomicrmw fadd ptr %{{.+}}, float %{{.+}} syncscope("agent") monotonic, align 4{{.*}}!atomic.ignore.denormal.mode !{{[0-9]+}}, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.no.remote.memory
__device__ float no_fg_add(float *p, float v) {
  [[clang::atomic(no_fine_grained_memory, ignore_denormal_mode)]] {
    return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, 4);
  }
}

// CIR-LABEL: @_Z9no_fg_addPdd
// CIR: cir.atomic.fetch add relaxed syncscope(hip_agent) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.double>, !cir.double) -> !cir.double {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z9no_fg_addPdd
// LLVM: atomicrmw fadd ptr %{{.+}}, double %{{.+}} syncscope("agent") monotonic
// LLVM-NOT: atomic.ignore.denormal.mode
__device__ double no_fg_add(double *p, double v) {
  [[clang::atomic(no_fine_grained_memory, ignore_denormal_mode)]] {
    return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, 4);
  }
}

// CIR-LABEL: @_Z12workgroup_ddPdd
// CIR: cir.atomic.fetch add relaxed syncscope(hip_workgroup) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.double>, !cir.double) -> !cir.double {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z12workgroup_ddPdd
// LLVM: atomicrmw fadd ptr %{{.+}}, double %{{.+}} syncscope("workgroup") monotonic
__device__ double workgroup_dd(double *p, double v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, 3);
}

// CIR-LABEL: @_Z12after_scopedPff
// CIR: cir.atomic.fetch add relaxed syncscope(hip_agent) fetch_first %{{.+}}, %{{.+}} : (!cir.ptr<!cir.float>, !cir.float) -> !cir.float {cir.amdgpu_no_fine_grained_memory, cir.amdgpu_no_remote_memory}
// LLVM-LABEL: @_Z12after_scopedPff
// LLVM: atomicrmw fadd ptr %{{.+}}, float %{{.+}} syncscope("agent") monotonic, align 4{{.*}}!amdgpu.no.fine.grained.memory
__device__ float after_scoped(float *p, float v) {
  [[clang::atomic(fine_grained_memory)]] { (void)0; }
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, 4);
}
