#include "Inputs/cuda.h"

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fclangir -emit-cir %s -o %t-amdgcn.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t-amdgcn.cir %s

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fclangir -emit-cir %s -o %t-spirv.cir
// RUN: FileCheck --check-prefix=CIR-SPV --input-file=%t-spirv.cir %s

// The VTT parameter points into the globals address space. The VTT address
// point keeps the address space of the VTT global and is cast at the call.

struct A { int a; };
struct B : virtual A { __device__ B() {} };
struct C : B { __device__ C() {} };

__device__ void f() { C c; }

// CIR-LABEL: cir.func {{.*}} @_ZN1CC1Ev(
// CIR:         %[[VTT:.*]] = cir.vtt.address_point @_ZTT1C, offset = 1 -> !cir.ptr<!cir.ptr<!void>>
// CIR:         %[[CAST:.*]] = cir.cast address_space %[[VTT]] : !cir.ptr<!cir.ptr<!void>> -> !cir.ptr<!cir.ptr<!void>, target_address_space(1)>
// CIR:         cir.call @_ZN1BC2Ev(%{{.*}}, %[[CAST]])

// CIR-LABEL: cir.func {{.*}} @_ZN1BC2Ev({{.*}}, %arg1: !cir.ptr<!cir.ptr<!void>, target_address_space(1)>
// CIR:         cir.vtt.address_point %{{.*}} : !cir.ptr<!cir.ptr<!void>, target_address_space(1)>, offset = 0 -> !cir.ptr<!cir.ptr<!void>, target_address_space(1)>

// LLVM-LABEL: define {{.*}}void @_ZN1CC1Ev(
// LLVM:         call void @_ZN1BC2Ev(ptr {{.*}}, ptr addrspace(1) noundef addrspacecast (ptr getelementptr inbounds nuw (i8, ptr @_ZTT1C, i64 8) to ptr addrspace(1)))

// LLVM-LABEL: define linkonce_odr void @_ZN1BC2Ev(ptr {{.*}}, ptr addrspace(1) noundef %{{.*}})

// CIR-SPV-LABEL: cir.func {{.*}} @_ZN1CC1Ev(
// CIR-SPV:         %[[VTT:.*]] = cir.vtt.address_point @_ZTT1C, offset = 1 -> !cir.ptr<!cir.ptr<!void, target_address_space(4)>>
// CIR-SPV:         %[[CAST:.*]] = cir.cast address_space %[[VTT]] : !cir.ptr<!cir.ptr<!void, target_address_space(4)>> -> !cir.ptr<!cir.ptr<!void, target_address_space(4)>, target_address_space(1)>
// CIR-SPV:         cir.call @_ZN1BC2Ev(%{{.*}}, %[[CAST]])

// CIR-SPV-LABEL: cir.func {{.*}} @_ZN1BC2Ev({{.*}}, %arg1: !cir.ptr<!cir.ptr<!void, target_address_space(4)>, target_address_space(1)>
// CIR-SPV:         cir.vtt.address_point %{{.*}} : !cir.ptr<!cir.ptr<!void, target_address_space(4)>, target_address_space(1)>, offset = 0 -> !cir.ptr<!cir.ptr<!void, target_address_space(4)>, target_address_space(1)>
