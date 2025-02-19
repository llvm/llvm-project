#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x hip -fhip-new-launch-api \
// RUN:            -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple=amdgcn-amd-amdhsa -x hip \
// RUN:            -fcuda-is-device -fhip-new-launch-api \
// RUN:              -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// Attribute for global_fn
// CIR-HOST: [[Kernel:#[a-zA-Z_0-9]+]] = {{.*}}#cir.cuda_kernel_name<_Z9global_fni>{{.*}}


__host__ void host_fn(int *a, int *b, int *c) {}
// CIR-HOST: cir.func @_Z7host_fnPiS_S_
// CIR-DEVICE-NOT: cir.func @_Z7host_fnPiS_S_

__device__ void device_fn(int* a, double b, float c) {}
// CIR-HOST-NOT: cir.func @_Z9device_fnPidf
// CIR-DEVICE: cir.func @_Z9device_fnPidf

__global__ void global_fn(int a) {}
// CIR-DEVICE: @_Z9global_fni

// CIR-HOST: cir.alloca {{.*}}"kernel_args"
// CIR-HOST: cir.call @__hipPopCallConfiguration

// Host access the global stub instead of the functiond evice stub.
// The stub has the mangled name of the function
// CIR-HOST: cir.get_global @_Z9global_fni
// CIR-HOST: cir.call @hipLaunchKernel
