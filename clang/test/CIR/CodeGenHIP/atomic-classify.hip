// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -fcuda-is-device \
// RUN: -emit-cir %s -o - | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fclangir -fcuda-is-device \
// RUN: -emit-llvm %s -o - | FileCheck --check-prefix=LLVM %s

#define __device__ __attribute__((device))

// CIR-LABEL: @_Z8hip_loadPi
// CIR: cir.load {{.*}} atomic(relaxed)
// LLVM-LABEL: @_Z8hip_loadPi
// LLVM: load atomic i32, ptr {{.*}} monotonic, align 4
__device__ int hip_load(int *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// CIR-LABEL: @_Z11opencl_loadPVU7_Atomici
// CIR: cir.load volatile {{.*}} atomic(relaxed)
// LLVM-LABEL: @_Z11opencl_loadPVU7_Atomici
// LLVM: load atomic volatile i32, ptr {{.*}} monotonic, align 4
__device__ int opencl_load(_Atomic(int) volatile *p) {
  return __opencl_atomic_load(p, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE);
}

// CIR-LABEL: @_Z9hip_storePii
// CIR: cir.store {{.*}} atomic(relaxed)
// LLVM-LABEL: @_Z9hip_storePii
// LLVM: store atomic i32 {{.*}} monotonic, align 4
__device__ void hip_store(int *p, int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// CIR-LABEL: @_Z12opencl_storePVU7_Atomicii
// CIR: cir.store volatile {{.*}} atomic(relaxed)
// LLVM-LABEL: @_Z12opencl_storePVU7_Atomicii
// LLVM: store atomic volatile i32 {{.*}} monotonic, align 4
__device__ void opencl_store(_Atomic(int) volatile *p, int v) {
  __opencl_atomic_store(p, v, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE);
}

// CIR-LABEL: @_Z12hip_exchangePii
// CIR: cir.atomic.xchg relaxed
// LLVM-LABEL: @_Z12hip_exchangePii
// LLVM: atomicrmw xchg ptr {{.*}} monotonic
__device__ int hip_exchange(int *p, int v) {
  return __hip_atomic_exchange(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// CIR-LABEL: @_Z15opencl_exchangePVU7_Atomicii
// CIR: cir.atomic.xchg relaxed
// LLVM-LABEL: @_Z15opencl_exchangePVU7_Atomicii
// LLVM: atomicrmw xchg ptr {{.*}} monotonic
__device__ int opencl_exchange(_Atomic(int) volatile *p, int v) {
  return __opencl_atomic_exchange(p, v, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE);
}

// CIR-LABEL: @_Z11hip_cmpxchgPiii
// CIR: cir.atomic.cmpxchg success(relaxed) failure(relaxed)
// LLVM-LABEL: @_Z11hip_cmpxchgPiii
// LLVM: cmpxchg ptr {{.*}} monotonic monotonic
__device__ bool hip_cmpxchg(int *p, int expected, int v) {
  return __hip_atomic_compare_exchange_strong(p, &expected, v, __ATOMIC_RELAXED,
                                               __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
}

// CIR-LABEL: @_Z16hip_cmpxchg_weakPiii
// CIR: cir.atomic.cmpxchg weak success(relaxed) failure(relaxed)
// LLVM-LABEL: @_Z16hip_cmpxchg_weakPiii
// LLVM: cmpxchg weak ptr {{.*}} monotonic monotonic
__device__ bool hip_cmpxchg_weak(int *p, int expected, int v) {
  return __hip_atomic_compare_exchange_weak(p, &expected, v, __ATOMIC_RELAXED,
                                             __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_AGENT);
}

// CIR-LABEL: @_Z14opencl_cmpxchgPVU7_AtomiciPii
// CIR: cir.atomic.cmpxchg success(relaxed) failure(relaxed)
// LLVM-LABEL: @_Z14opencl_cmpxchgPVU7_AtomiciPii
// LLVM: cmpxchg volatile ptr {{.*}} monotonic monotonic
__device__ bool opencl_cmpxchg(_Atomic(int) volatile *p, int *expected, int v) {
  return __opencl_atomic_compare_exchange_strong(p, expected, v, __ATOMIC_RELAXED,
                                                  __ATOMIC_RELAXED,
                                                  __OPENCL_MEMORY_SCOPE_DEVICE);
}

// CIR-LABEL: @_Z19opencl_cmpxchg_weakPVU7_AtomiciPii
// CIR: cir.atomic.cmpxchg weak success(relaxed) failure(relaxed)
// LLVM-LABEL: @_Z19opencl_cmpxchg_weakPVU7_AtomiciPii
// LLVM: cmpxchg weak volatile ptr {{.*}} monotonic monotonic
__device__ bool opencl_cmpxchg_weak(_Atomic(int) volatile *p, int *expected, int v) {
  return __opencl_atomic_compare_exchange_weak(p, expected, v, __ATOMIC_RELAXED,
                                                __ATOMIC_RELAXED,
                                                __OPENCL_MEMORY_SCOPE_DEVICE);
}

// CIR-LABEL: @_Z13hip_fetch_addPii
// CIR: cir.atomic.fetch add relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z13hip_fetch_addPii
// LLVM: atomicrmw add ptr {{.*}} monotonic
__device__ int hip_fetch_add(int *p, int v) {
  return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// CIR-LABEL: @_Z16opencl_fetch_addPVU7_Atomicii
// CIR: cir.atomic.fetch add relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z16opencl_fetch_addPVU7_Atomicii
// LLVM: atomicrmw add ptr {{.*}} monotonic
__device__ int opencl_fetch_add(_Atomic(int) volatile *p, int v) {
  return __opencl_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE);
}

// CIR-LABEL: @_Z13hip_fetch_subPii
// CIR: cir.atomic.fetch sub relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z13hip_fetch_subPii
// LLVM: atomicrmw sub ptr {{.*}} monotonic
__device__ int hip_fetch_sub(int *p, int v) {
  return __hip_atomic_fetch_sub(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// CIR-LABEL: @_Z16opencl_fetch_subPVU7_Atomicii
// CIR: cir.atomic.fetch sub relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z16opencl_fetch_subPVU7_Atomicii
// LLVM: atomicrmw sub ptr {{.*}} monotonic
__device__ int opencl_fetch_sub(_Atomic(int) volatile *p, int v) {
  return __opencl_atomic_fetch_sub(p, v, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE);
}

// CIR-LABEL: @_Z13hip_fetch_minPii
// CIR: cir.atomic.fetch min relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z13hip_fetch_minPii
// LLVM: atomicrmw min ptr {{.*}} monotonic
__device__ int hip_fetch_min(int *p, int v) {
  return __hip_atomic_fetch_min(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// CIR-LABEL: @_Z16opencl_fetch_minPVU7_Atomicii
// CIR: cir.atomic.fetch min relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z16opencl_fetch_minPVU7_Atomicii
// LLVM: atomicrmw min ptr {{.*}} monotonic
__device__ int opencl_fetch_min(_Atomic(int) volatile *p, int v) {
  return __opencl_atomic_fetch_min(p, v, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE);
}

// CIR-LABEL: @_Z13hip_fetch_maxPii
// CIR: cir.atomic.fetch max relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z13hip_fetch_maxPii
// LLVM: atomicrmw max ptr {{.*}} monotonic
__device__ int hip_fetch_max(int *p, int v) {
  return __hip_atomic_fetch_max(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// CIR-LABEL: @_Z16opencl_fetch_maxPVU7_Atomicii
// CIR: cir.atomic.fetch max relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z16opencl_fetch_maxPVU7_Atomicii
// LLVM: atomicrmw max ptr {{.*}} monotonic
__device__ int opencl_fetch_max(_Atomic(int) volatile *p, int v) {
  return __opencl_atomic_fetch_max(p, v, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE);
}

// CIR-LABEL: @_Z13hip_fetch_andPii
// CIR: cir.atomic.fetch and relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z13hip_fetch_andPii
// LLVM: atomicrmw and ptr {{.*}} monotonic
__device__ int hip_fetch_and(int *p, int v) {
  return __hip_atomic_fetch_and(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// CIR-LABEL: @_Z16opencl_fetch_andPVU7_Atomicii
// CIR: cir.atomic.fetch and relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z16opencl_fetch_andPVU7_Atomicii
// LLVM: atomicrmw and ptr {{.*}} monotonic
__device__ int opencl_fetch_and(_Atomic(int) volatile *p, int v) {
  return __opencl_atomic_fetch_and(p, v, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE);
}

// CIR-LABEL: @_Z12hip_fetch_orPii
// CIR: cir.atomic.fetch or relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z12hip_fetch_orPii
// LLVM: atomicrmw or ptr {{.*}} monotonic
__device__ int hip_fetch_or(int *p, int v) {
  return __hip_atomic_fetch_or(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// CIR-LABEL: @_Z15opencl_fetch_orPVU7_Atomicii
// CIR: cir.atomic.fetch or relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z15opencl_fetch_orPVU7_Atomicii
// LLVM: atomicrmw or ptr {{.*}} monotonic
__device__ int opencl_fetch_or(_Atomic(int) volatile *p, int v) {
  return __opencl_atomic_fetch_or(p, v, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE);
}

// CIR-LABEL: @_Z13hip_fetch_xorPii
// CIR: cir.atomic.fetch xor relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z13hip_fetch_xorPii
// LLVM: atomicrmw xor ptr {{.*}} monotonic
__device__ int hip_fetch_xor(int *p, int v) {
  return __hip_atomic_fetch_xor(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// CIR-LABEL: @_Z16opencl_fetch_xorPVU7_Atomicii
// CIR: cir.atomic.fetch xor relaxed {{.*}} fetch_first
// LLVM-LABEL: @_Z16opencl_fetch_xorPVU7_Atomicii
// LLVM: atomicrmw xor ptr {{.*}} monotonic
__device__ int opencl_fetch_xor(_Atomic(int) volatile *p, int v) {
  return __opencl_atomic_fetch_xor(p, v, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_DEVICE);
}
