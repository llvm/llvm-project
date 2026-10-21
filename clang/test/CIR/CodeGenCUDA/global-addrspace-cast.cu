#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fclangir -emit-llvm %s -o %t-cir-amdgcn.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir-amdgcn.ll %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -emit-llvm %s -o %t-amdgcn.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t-amdgcn.ll %s

// The address of a global whose address space differs from its declared type
// is cast to the declared (generic) address space where it is formed.

__device__ int g;
__device__ int arr[4];
__shared__ int sh;
extern __shared__ int dyn[];

__device__ int *addr_of_global() { return &g; }

// CIR-LABEL: cir.func {{.*}}@_Z14addr_of_globalv
// CIR: %[[G:.*]] = cir.get_global @g : !cir.ptr<{{.*}}, target_address_space(1)>
// CIR: cir.cast address_space %[[G]] : !cir.ptr<{{.*}}, target_address_space(1)> -> !cir.ptr<{{.*}}>

// LLVM-LABEL: @_Z14addr_of_globalv
// LLVM: store ptr addrspacecast (ptr addrspace(1) @g to ptr)
// OGCG-LABEL: @_Z14addr_of_globalv
// OGCG: ret ptr addrspacecast (ptr addrspace(1) @g to ptr)

__device__ int *array_decay() { return arr; }

// CIR-LABEL: cir.func {{.*}}@_Z11array_decayv
// CIR: %[[G:.*]] = cir.get_global @arr : !cir.ptr<{{.*}}, target_address_space(1)>
// CIR: cir.cast address_space %[[G]] : !cir.ptr<{{.*}}, target_address_space(1)> -> !cir.ptr<{{.*}}>

// LLVM-LABEL: @_Z11array_decayv
// LLVM: store ptr addrspacecast (ptr addrspace(1) @arr to ptr)
// OGCG-LABEL: @_Z11array_decayv
// OGCG: ret ptr addrspacecast (ptr addrspace(1) @arr to ptr)

__device__ int &bind_ref() { return g; }

// CIR-LABEL: cir.func {{.*}}@_Z8bind_refv
// CIR: %[[G:.*]] = cir.get_global @g : !cir.ptr<{{.*}}, target_address_space(1)>
// CIR: cir.cast address_space %[[G]] : !cir.ptr<{{.*}}, target_address_space(1)> -> !cir.ptr<{{.*}}>

// LLVM-LABEL: @_Z8bind_refv
// LLVM: store ptr addrspacecast (ptr addrspace(1) @g to ptr)
// OGCG-LABEL: @_Z8bind_refv
// OGCG: ret ptr addrspacecast (ptr addrspace(1) @g to ptr)

__device__ int *addr_of_shared() { return &sh; }

// CIR-LABEL: cir.func {{.*}}@_Z14addr_of_sharedv
// CIR: %[[G:.*]] = cir.get_global @sh : !cir.ptr<{{.*}}, target_address_space(3)>
// CIR: cir.cast address_space %[[G]] : !cir.ptr<{{.*}}, target_address_space(3)> -> !cir.ptr<{{.*}}>

// LLVM-LABEL: @_Z14addr_of_sharedv
// LLVM: store ptr addrspacecast (ptr addrspace(3) @sh to ptr)
// OGCG-LABEL: @_Z14addr_of_sharedv
// OGCG: ret ptr addrspacecast (ptr addrspace(3) @sh to ptr)

__device__ int *dynamic_shared() { return dyn; }

// CIR-LABEL: cir.func {{.*}}@_Z14dynamic_sharedv
// CIR: %[[G:.*]] = cir.get_global @dyn : !cir.ptr<{{.*}}, target_address_space(3)>
// CIR: cir.cast address_space %[[G]] : !cir.ptr<{{.*}}, target_address_space(3)> -> !cir.ptr<{{.*}}>

// LLVM-LABEL: @_Z14dynamic_sharedv
// LLVM: store ptr addrspacecast (ptr addrspace(3) @dyn to ptr)
// OGCG-LABEL: @_Z14dynamic_sharedv
// OGCG: ret ptr addrspacecast (ptr addrspace(3) @dyn to ptr)

__device__ int *addr_of_static_shared() {
  __shared__ int s;
  return &s;
}

// CIR-LABEL: cir.func {{.*}}@_Z21addr_of_static_sharedv
// CIR: %[[G:.*]] = cir.get_global @_ZZ21addr_of_static_sharedvE1s : !cir.ptr<{{.*}}, target_address_space(3)>
// CIR: cir.cast address_space %[[G]] : !cir.ptr<{{.*}}, target_address_space(3)> -> !cir.ptr<{{.*}}>

// LLVM-LABEL: @_Z21addr_of_static_sharedv
// LLVM: store ptr addrspacecast (ptr addrspace(3) @_ZZ21addr_of_static_sharedvE1s to ptr)
// OGCG-LABEL: @_Z21addr_of_static_sharedv
// OGCG: ret ptr addrspacecast (ptr addrspace(3) @_ZZ21addr_of_static_sharedvE1s to ptr)
