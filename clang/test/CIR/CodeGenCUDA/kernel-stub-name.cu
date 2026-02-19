// Based on clang/test/CodeGenCUDA/kernel-stub-name.cu.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-sdk-version=9.2 \
// RUN:   -emit-cir %s -x cuda -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "Inputs/cuda.h"

// CHECK: cir.func {{.*}} @[[CSTUB:__device_stub__ckernel]]()
// CHECK: cir.return
// CHECK-NEXT: }
extern "C" __global__ void ckernel() {}

// CHECK: cir.func {{.*}} @_ZN2ns23__device_stub__nskernelEv()
namespace ns {
__global__ void nskernel() {}
} // namespace ns

// CHECK: cir.func {{.*}} @_Z25__device_stub__kernelfuncIiEvv()
template <class T>
__global__ void kernelfunc() {}
template __global__ void kernelfunc<int>();
