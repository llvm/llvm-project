// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm -o - -fcuda-is-device %s \
// RUN:   | FileCheck --check-prefix=DEFAULT %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm -o - -fcuda-is-device \
// RUN:   -fno-private-atomics-undefined %s \
// RUN:   | FileCheck --check-prefix=NO-PRIV-UNDEF %s

#include "Inputs/cuda.h"

// DEFAULT-LABEL: define {{.*}}@_Z18test_opencl_atomicPU7_Atomici
// DEFAULT: atomicrmw add
// DEFAULT-SAME: !noalias.addrspace
//
// NO-PRIV-UNDEF-LABEL: define {{.*}}@_Z18test_opencl_atomicPU7_Atomici
// NO-PRIV-UNDEF: !noalias.addrspace
__device__ void test_opencl_atomic(_Atomic int *a) {
  __opencl_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __OPENCL_MEMORY_SCOPE_WORK_GROUP);
}

// DEFAULT-LABEL: define {{.*}}@_Z18test_scoped_atomicPi
// DEFAULT: atomicrmw add
// DEFAULT-SAME: !noalias.addrspace
//
// NO-PRIV-UNDEF-NOT: !noalias.addrspace
// NO-PRIV-UNDEF: ret void
__device__ void test_scoped_atomic(int *a) {
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}
