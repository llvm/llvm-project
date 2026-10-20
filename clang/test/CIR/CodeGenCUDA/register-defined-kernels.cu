// RUN: echo -n "GPU binary would be here." > %t
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-llvm %s -x hip -fhip-new-launch-api -fcuda-include-gpubinary %t -o %t-hip-cir.ll
// RUN: FileCheck --input-file=%t-hip-cir.ll %s --check-prefix=HIP
// RUN: FileCheck --input-file=%t-hip-cir.ll %s --check-prefix=NOREG-HIP
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -x hip -fhip-new-launch-api -fcuda-include-gpubinary %t -o %t-hip-ogcg.ll
// RUN: FileCheck --input-file=%t-hip-ogcg.ll %s --check-prefix=HIP
// RUN: FileCheck --input-file=%t-hip-ogcg.ll %s --check-prefix=NOREG-HIP
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-llvm %s -x cuda -target-sdk-version=12.3 -fcuda-include-gpubinary %t -o %t-cuda-cir.ll
// RUN: FileCheck --input-file=%t-cuda-cir.ll %s --check-prefix=CUDA
// RUN: FileCheck --input-file=%t-cuda-cir.ll %s --check-prefix=NOREG-CUDA
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -x cuda -target-sdk-version=12.3 -fcuda-include-gpubinary %t -o %t-cuda-ogcg.ll
// RUN: FileCheck --input-file=%t-cuda-ogcg.ll %s --check-prefix=CUDA
// RUN: FileCheck --input-file=%t-cuda-ogcg.ll %s --check-prefix=NOREG-CUDA

#include "Inputs/cuda.h"

template <int N> __global__ void kern(int *p) { p[0] = N; }

// Instantiated in another translation unit.
extern template __global__ void kern<1>(int *);
// Instantiated here.
template __global__ void kern<2>(int *);

// Defined in another translation unit.
__global__ void decl_only(int *);
// Defined after its launch.
__global__ void fwd(int *);
// Defined but not launched.
__global__ void unused(int *p) { p[0] = 0; }

void launch(int *p) {
  kern<1><<<1, 1>>>(p);
  kern<2><<<1, 1>>>(p);
  // Implicitly instantiated here.
  kern<3><<<1, 1>>>(p);
  decl_only<<<1, 1>>>(p);
  fwd<<<1, 1>>>(p);
}

__global__ void fwd(int *p) { p[0] = 4; }

// NOREG-HIP-NOT:  @__hipRegisterFunction({{.*}}@_Z4kernILi1EEvPi
// NOREG-HIP-NOT:  @__hipRegisterFunction({{.*}}@_Z9decl_onlyPi
// NOREG-CUDA-NOT: @__cudaRegisterFunction({{.*}}@_Z19__device_stub__kernILi1EEvPi
// NOREG-CUDA-NOT: @__cudaRegisterFunction({{.*}}@_Z24__device_stub__decl_onlyPi

// HIP-LABEL: define internal void @__hip_register_globals(
// HIP-DAG:     call {{.*}}@__hipRegisterFunction({{.*}}@_Z4kernILi2EEvPi
// HIP-DAG:     call {{.*}}@__hipRegisterFunction({{.*}}@_Z4kernILi3EEvPi
// HIP-DAG:     call {{.*}}@__hipRegisterFunction({{.*}}@_Z3fwdPi
// HIP-DAG:     call {{.*}}@__hipRegisterFunction({{.*}}@_Z6unusedPi
// HIP:         ret void

// CUDA-LABEL: define internal void @__cuda_register_globals(
// CUDA-DAG:     call {{.*}}@__cudaRegisterFunction({{.*}}@_Z19__device_stub__kernILi2EEvPi
// CUDA-DAG:     call {{.*}}@__cudaRegisterFunction({{.*}}@_Z19__device_stub__kernILi3EEvPi
// CUDA-DAG:     call {{.*}}@__cudaRegisterFunction({{.*}}@_Z18__device_stub__fwdPi
// CUDA-DAG:     call {{.*}}@__cudaRegisterFunction({{.*}}@_Z21__device_stub__unusedPi
// CUDA:         ret void
