// RUN: echo "GPU binary would be here" > %t

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - -x hip\
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s --check-prefixes=CHECK

#include "Inputs/cuda.h"

extern "C" __global__ void ckernel() {}

namespace ns {
__global__ void nskernel() {}
} // namespace ns

template<class T>
__global__ void kernelfunc() {}

__global__ void kernel_decl();

// Device side kernel names

// CHECK: @[[CKERN:[0-9]*]] = {{.*}} c"ckernel\00"
// CHECK: @[[NSKERN:[0-9]*]] = {{.*}} c"_ZN2ns8nskernelEv\00"
// CHECK: @[[TKERN:[0-9]*]] = {{.*}} c"_Z10kernelfuncIiEvv\00"

// Non-template kernel stub functions

// CHECK: define{{.*}}@[[CSTUB:__device_stub__ckernel]]
// CHECK: call{{.*}}@hipLaunchByPtr{{.*}}@[[CSTUB]]
// CHECK: define{{.*}}@[[NSSTUB:_ZN2ns23__device_stub__nskernelEv]]
// CHECK: call{{.*}}@hipLaunchByPtr{{.*}}@[[NSSTUB]]

// CHECK-LABEL: define{{.*}}@_Z8hostfuncv()
// CHECK: call void @[[CSTUB]]()
// CHECK: call void @[[NSSTUB]]()
// CHECK: call void @[[TSTUB:_Z25__device_stub__kernelfuncIiEvv]]()
// CHECK: call void @[[DSTUB:_Z26__device_stub__kernel_declv]]()
void hostfunc(void) {
  ckernel<<<1, 1>>>();
  ns::nskernel<<<1, 1>>>();
  kernelfunc<int><<<1, 1>>>();
  kernel_decl<<<1, 1>>>();
}

// Template kernel stub functions

// CHECK: define{{.*}}@[[TSTUB]]
// CHECK: call{{.*}}@hipLaunchByPtr{{.*}}@[[TSTUB]]

// CHECK: declare{{.*}}@[[DSTUB]]

// CHECK-LABEL: define{{.*}}@__hip_register_globals
// CHECK: call{{.*}}@__hipRegisterFunction{{.*}}@[[CSTUB]]{{.*}}@[[CKERN]]
// CHECK: call{{.*}}@__hipRegisterFunction{{.*}}@[[NSSTUB]]{{.*}}@[[NSKERN]]
// CHECK: call{{.*}}@__hipRegisterFunction{{.*}}@[[TSTUB]]{{.*}}@[[TKERN]]
