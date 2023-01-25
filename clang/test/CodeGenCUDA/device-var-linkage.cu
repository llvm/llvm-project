// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device \
// RUN:   -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=DEV,NORDC %s
// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=DEV,RDC %s
// RUN: %clang_cc1 -triple x86_64-unknown-gnu-linux \
// RUN:   -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=HOST,NORDC-H %s
// RUN: %clang_cc1 -triple x86_64-unknown-gnu-linux \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck -check-prefixes=HOST,RDC-H %s
// RUN: %clang_cc1 -triple nvptx -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefixes=CUDA %s

#include "Inputs/cuda.h"

// DEV-DAG: @v1 = addrspace(1) externally_initialized global i32 0
// NORDC-H-DAG: @v1 = internal global i32 undef
// RDC-H-DAG: @v1 = global i32 undef
__device__ int v1;
// DEV-DAG: @v2 = addrspace(4) externally_initialized global i32 0
// NORDC-H-DAG: @v2 = internal global i32 undef
// RDC-H-DAG: @v2 = global i32 undef
__constant__ int v2;
// DEV-DAG: @v3 = addrspace(1) externally_initialized global ptr addrspace(1) null
// NORDC-H-DAG: @v3 = internal externally_initialized global ptr null
// RDC-H-DAG: @v3 = externally_initialized global ptr null
#if __HIP__
__managed__ int v3;
#endif

// DEV-DAG: @ev1 = external addrspace(1) global i32
// HOST-DAG: @ev1 = external global i32
extern __device__ int ev1;
// DEV-DAG: @ev2 = external addrspace(4) global i32
// HOST-DAG: @ev2 = external global i32
extern __constant__ int ev2;
// DEV-DAG: @ev3 = external addrspace(1) externally_initialized global ptr addrspace(1)
// HOST-DAG: @ev3 = external externally_initialized global ptr
#if __HIP__
extern __managed__ int ev3;
#endif

// NORDC-DAG: @_ZL3sv1 = addrspace(1) externally_initialized global i32 0
// RDC-DAG: @_ZL3sv1.static.[[HASH:.*]] = addrspace(1) externally_initialized global i32 0
// HOST-DAG: @_ZL3sv1 = internal global i32 undef
// CUDA-DAG: @_ZL3sv1__static__[[HASH:.*]] = addrspace(1) externally_initialized global i32 0
static __device__ int sv1;
// NORDC-DAG: @_ZL3sv2 = addrspace(4) externally_initialized global i32 0
// RDC-DAG: @_ZL3sv2.static.[[HASH]] = addrspace(4) externally_initialized global i32 0
// HOST-DAG: @_ZL3sv2 = internal global i32 undef
// CUDA-DAG: @_ZL3sv2__static__[[HASH]] = addrspace(4) externally_initialized global i32 0
static __constant__ int sv2;
// NORDC-DAG: @_ZL3sv3 = addrspace(1) externally_initialized global ptr addrspace(1) null
// RDC-DAG: @_ZL3sv3.static.[[HASH]] = addrspace(1) externally_initialized global ptr addrspace(1) null
// HOST-DAG: @_ZL3sv3 = internal externally_initialized global ptr null
#if __HIP__
static __managed__ int sv3;
#endif

__device__ __host__ int work(int *x);

__device__ __host__ int fun1() {
  return work(&ev1) + work(&ev2) + work(&sv1) + work(&sv2)
#if __HIP__
    + work(&ev3) + work(&sv3)
#endif
    ;
}

// HOST: hipRegisterVar({{.*}}@v1
// HOST: hipRegisterVar({{.*}}@v2
// HOST: hipRegisterManagedVar({{.*}}@v3
// HOST-NOT: hipRegisterVar({{.*}}@ev1
// HOST-NOT: hipRegisterVar({{.*}}@ev2
// HOST-NOT: hipRegisterManagedVar({{.*}}@ev3
// HOST: hipRegisterVar({{.*}}@_ZL3sv1
// HOST: hipRegisterVar({{.*}}@_ZL3sv2
// HOST: hipRegisterManagedVar({{.*}}@_ZL3sv3
