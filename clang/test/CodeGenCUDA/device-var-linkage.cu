// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device \
// RUN:   -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=DEV,HIP-D,NORDC,HIP-NORDC %s
// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=DEV,HIP-D %s
// RUN: %clang_cc1 -triple x86_64-unknown-gnu-linux \
// RUN:   -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=HOST,HIP-H,NORDC-H %s
// RUN: %clang_cc1 -triple x86_64-unknown-gnu-linux \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=HOST,HIP-H,RDC-H %s

// RUN: %clang_cc1 -triple nvptx -fcuda-is-device \
// RUN:   -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefixes=DEV,CUDA-D,NORDC,CUDA-NORDC %s
// RUN: %clang_cc1 -triple nvptx -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefixes=DEV,CUDA-D %s

// RUN: %clang_cc1 -triple x86_64-unknown-gnu-linux \
// RUN:   -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefixes=HOST,NORDC-H %s
// RUN: %clang_cc1 -triple x86_64-unknown-gnu-linux \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm -o - -x cuda %s \
// RUN:   | FileCheck -check-prefixes=HOST,RDC-H %s

#include "Inputs/cuda.h"

// DEV-DAG: @v1 = addrspace(1) externally_initialized global i32 0
// NORDC-H-DAG: @v1 = internal global i32 undef
// RDC-H-DAG: @v1 = global i32 undef
__device__ int v1;
// DEV-DAG: @v2 = addrspace(4) externally_initialized constant i32 0
// NORDC-H-DAG: @v2 = internal global i32 undef
// RDC-H-DAG: @v2 = global i32 undef
__constant__ int v2;
// HIP-D-DAG: @v3 = addrspace(1) externally_initialized global ptr addrspace(1) null
// CUDA-D-DAG: @v3 = addrspace(1) externally_initialized global i32 0, align 4
// NORDC-H-DAG: @v3 = internal externally_initialized global ptr null
// RDC-H-DAG: @v3 = externally_initialized global ptr null
__managed__ int v3;

// DEV-DAG: @ev1 = external addrspace(1) global i32
// HOST-DAG: @ev1 = external global i32
extern __device__ int ev1;
// DEV-DAG: @ev2 = external addrspace(4) global i32
// HOST-DAG: @ev2 = external global i32
extern __constant__ int ev2;
// HIP-D-DAG: @ev3 = external addrspace(1) externally_initialized global ptr addrspace(1)
// CUDA-D-DAG: @ev3 = external addrspace(1) global i32, align 4
// HOST-DAG: @ev3 = external externally_initialized global ptr
extern __managed__ int ev3;

// NORDC-DAG: @_ZL3sv1 = addrspace(1) externally_initialized global i32 0
// HIP-RDC-DAG: @_ZL3sv1.static.[[HASH:.*]] = addrspace(1) externally_initialized global i32 0
// CUDA-RDC-DAG: @_ZL3sv1__static__[[HASH:.*]] = addrspace(1) externally_initialized global i32 0
// HOST-DAG: @_ZL3sv1 = internal global i32 undef
static __device__ int sv1;
// NORDC-DAG: @_ZL3sv2 = addrspace(4) externally_initialized constant i32 0
// HIP-RDC-DAG: @_ZL3sv2.static.[[HASH]] = addrspace(4) externally_initialized constant i32 0
// CUDA-RDC-DAG: @_ZL3sv2__static__[[HASH]] = addrspace(4) externally_initialized constant i32 0
// HOST-DAG: @_ZL3sv2 = internal global i32 undef
static __constant__ int sv2;
// HIP-NORDC-DAG: @_ZL3sv3 = addrspace(1) externally_initialized global ptr addrspace(1) null
// CUDA-NORDC-DAG: @_ZL3sv3 = addrspace(1) externally_initialized global i32 0, align 4
// HIP-RDC-DAG: @_ZL3sv3.static.[[HASH]] = addrspace(1) externally_initialized global ptr addrspace(1) null
// CUDA-RDC-DAG: @_ZL3sv3__static__[[HASH]] = addrspace(1) externally_initialized global i32 0, align 4
// HOST-DAG: @_ZL3sv3 = internal externally_initialized global ptr null
static __managed__ int sv3;

__device__ __host__ int work(int *x);

__device__ __host__ int fun1() {
  return work(&ev1) + work(&ev2) + work(&sv1) + work(&sv2) +
         work(&ev3) + work(&sv3);
}

// HIP-H: hipRegisterVar({{.*}}@v1
// HIP-H: hipRegisterVar({{.*}}@v2
// HIP-H: hipRegisterManagedVar({{.*}}@v3
// HIP-H-NOT: hipRegisterVar({{.*}}@ev1
// HIP-H-NOT: hipRegisterVar({{.*}}@ev2
// HIP-H-NOT: hipRegisterManagedVar({{.*}}@ev3
// HIP-H: hipRegisterVar({{.*}}@_ZL3sv1
// HIP-H: hipRegisterVar({{.*}}@_ZL3sv2
// HIP-H: hipRegisterManagedVar({{.*}}@_ZL3sv3
