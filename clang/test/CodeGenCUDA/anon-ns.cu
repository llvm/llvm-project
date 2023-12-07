// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -std=c++17 -fgpu-rdc \
// RUN:   -emit-llvm -o - -x hip %s > %t.dev

// RUN: %clang_cc1 -triple x86_64-gnu-linux -cuid=abc \
// RUN:   -aux-triple amdgcn-amd-amdhsa -std=c++17 -fgpu-rdc \
// RUN:   -emit-llvm -o - -x hip %s > %t.host

// RUN: cat %t.dev %t.host | FileCheck -check-prefixes=HIP,COMMON %s
// RUN: cat %t.dev %t.host | FileCheck -check-prefixes=COMNEG %s

// RUN: echo "GPU binary" > %t.fatbin

// RUN: %clang_cc1 -triple nvptx -fcuda-is-device -cuid=abc \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -std=c++17 -fgpu-rdc \
// RUN:   -emit-llvm -o - %s > %t.dev

// RUN: %clang_cc1 -triple x86_64-gnu-linux -cuid=abc \
// RUN:   -aux-triple nvptx -std=c++17 -fgpu-rdc -fcuda-include-gpubinary %t.fatbin \
// RUN:   -emit-llvm -o - %s > %t.host

// RUN: cat %t.dev %t.host | FileCheck -check-prefixes=CUDA,COMMON %s
// RUN: cat %t.dev %t.host | FileCheck -check-prefixes=COMNEG %s

#include "Inputs/cuda.h"

// HIP-DAG: define weak_odr {{.*}}void @[[KERN:_ZN12_GLOBAL__N_16kernelEv\.intern\.b04fd23c98500190]](
// HIP-DAG: define weak_odr {{.*}}void @[[KTX:_Z2ktIN12_GLOBAL__N_11XEEvT_\.intern\.b04fd23c98500190]](
// HIP-DAG: define weak_odr {{.*}}void @[[KTL:_Z2ktIN12_GLOBAL__N_1UlvE_EEvT_\.intern\.b04fd23c98500190]](
// HIP-DAG: @[[VM:_ZN12_GLOBAL__N_12vmE\.static\.b04fd23c98500190]] = addrspace(1) externally_initialized global
// HIP-DAG: @[[VC:_ZN12_GLOBAL__N_12vcE\.static\.b04fd23c98500190]] = addrspace(4) externally_initialized global
// HIP-DAG: @[[VT:_Z2vtIN12_GLOBAL__N_11XEE\.static\.b04fd23c98500190]] = addrspace(1) externally_initialized global

// CUDA-DAG: define weak_odr {{.*}}void @[[KERN:_ZN12_GLOBAL__N_16kernelEv__intern__b04fd23c98500190]](
// CUDA-DAG: define weak_odr {{.*}}void @[[KTX:_Z2ktIN12_GLOBAL__N_11XEEvT___intern__b04fd23c98500190]](
// CUDA-DAG: define weak_odr {{.*}}void @[[KTL:_Z2ktIN12_GLOBAL__N_1UlvE_EEvT___intern__b04fd23c98500190]](
// CUDA-DAG: @[[VC:_ZN12_GLOBAL__N_12vcE__static__b04fd23c98500190]] = addrspace(4) externally_initialized global
// CUDA-DAG: @[[VT:_Z2vtIN12_GLOBAL__N_11XEE__static__b04fd23c98500190]] = addrspace(1) externally_initialized global

// COMMON-DAG: @_ZN12_GLOBAL__N_12vdE = internal addrspace(1) global
// COMNEG-NOT: @{{.*}} = {{.*}} c"_ZN12_GLOBAL__N_12vdE{{.*}}\00"

// HIP-DAG: @llvm.compiler.used = {{.*}}@[[VM]]{{.*}}@[[VT]]{{.*}}@[[VC]]
// CUDA-DAG: @llvm.compiler.used = {{.*}}@[[VT]]{{.*}}@[[VC]]

// COMMON-DAG: @[[KERNSTR:.*]] = {{.*}} c"[[KERN]]\00"
// COMMON-DAG: @[[KTXSTR:.*]] = {{.*}} c"[[KTX]]\00"
// COMMON-DAG: @[[KTLSTR:.*]] = {{.*}} c"[[KTL]]\00"
// HIP-DAG: @[[VMSTR:.*]] = {{.*}} c"[[VM]]\00"
// COMMON-DAG: @[[VCSTR:.*]] = {{.*}} c"[[VC]]\00"
// COMMON-DAG: @[[VTSTR:.*]] = {{.*}} c"[[VT]]\00"

// COMMON-DAG: call i32 @__{{.*}}RegisterFunction({{.*}}@[[KERNSTR]]
// COMMON-DAG: call i32 @__{{.*}}RegisterFunction({{.*}}@[[KTXSTR]]
// COMMON-DAG: call i32 @__{{.*}}RegisterFunction({{.*}}@[[KTLSTR]]
// HIP-DAG: call void @__{{.*}}RegisterManagedVar({{.*}}@[[VMSTR]]
// COMMON-DAG: call void @__{{.*}}RegisterVar({{.*}}@[[VCSTR]]
// COMMON-DAG: call void @__{{.*}}RegisterVar({{.*}}@[[VTSTR]]

template <typename T>
__global__ void kt(T x) {}

template <typename T>
__device__ T vt;

namespace {
  struct X {};
  X x;
  auto lambda = [](){};
#if __HIP__
  __managed__ int vm = 1;
#endif
  __constant__ int vc = 2;

  // C should not be externalized since it is used by device code only.
  __device__ int vd = 3;
  __global__ void kernel() { vd = 4; }
}

template<typename T>
void getSymbol(T *x) {}

void test() {
  kernel<<<1, 1>>>();

  kt<<<1, 1>>>(x);

  kt<<<1, 1>>>(lambda);

  // A, B, and tempVar<X> should be externalized since they are
  // used by host code.
#if __HIP__
  getSymbol(&vm);
#endif
  getSymbol(&vc);
  getSymbol(&vt<X>);
}
