// RUN: echo "GPU binary would be here" > %t

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -O0 \
// RUN:   -fcuda-include-gpubinary %t -debug-info-kind=limited \
// RUN:   -o - -x hip | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -O0 \
// RUN:   -fcuda-include-gpubinary %t -debug-info-kind=limited \
// RUN:   -o - -x hip -fcuda-is-device | FileCheck -check-prefix=DEV %s

#include "Inputs/cuda.h"

extern "C" __global__ void ckernel(int *a) {
  *a = 1;
}

// Device side kernel names
// CHECK: @[[CKERN:[0-9]*]] = {{.*}} c"ckernel\00"

// DEV: define {{.*}}@ckernel{{.*}}!dbg
// DEV:  store {{.*}}!dbg
// DEV:  ret {{.*}}!dbg

// CHECK-NOT: define {{.*}}@__device_stub__ckernel{{.*}}!dbg
// CHECK: define {{.*}}@[[CSTUB:__device_stub__ckernel]]
// CHECK-NOT: call {{.*}}@hipLaunchByPtr{{.*}}!dbg
// CHECK: call {{.*}}@hipLaunchByPtr{{.*}}@[[CSTUB]]
// CHECK-NOT: ret {{.*}}!dbg

// CHECK-LABEL: define {{.*}}@_Z8hostfuncPi{{.*}}!dbg
// CHECK: call void @[[CSTUB]]{{.*}}!dbg
void hostfunc(int *a) {
  ckernel<<<1, 1>>>(a);
}
