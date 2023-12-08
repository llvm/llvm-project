// RUN: echo "GPU binary would be here" > %t

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -O0 \
// RUN:   -fcuda-include-gpubinary %t -debug-info-kind=limited \
// RUN:   -o - -x hip | FileCheck -check-prefixes=CHECK,O0 %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -O0 \
// RUN:   -fcuda-include-gpubinary %t -debug-info-kind=limited \
// RUN:   -o - -x hip -fcuda-is-device | FileCheck -check-prefix=DEV %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -O0 \
// RUN:   -fcuda-include-gpubinary %t -debug-info-kind=limited \
// RUN:   -o - -x hip -debugger-tuning=gdb -dwarf-version=4 \
// RUN:   | FileCheck -check-prefixes=CHECK,O0 %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -O0 \
// RUN:   -fcuda-include-gpubinary %t -debug-info-kind=limited \
// RUN:   -o - -x hip -debugger-tuning=gdb -dwarf-version=4 \
// RUN:   -fcuda-is-device | FileCheck -check-prefix=DEV %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -O3 \
// RUN:   -fcuda-include-gpubinary %t -debug-info-kind=limited \
// RUN:   -o - -x hip -debugger-tuning=gdb -dwarf-version=4 | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm %s -O3 \
// RUN:   -fcuda-include-gpubinary %t -debug-info-kind=limited \
// RUN:   -o - -x hip -debugger-tuning=gdb -dwarf-version=4 \
// RUN:   -fcuda-is-device | FileCheck -check-prefix=DEV %s

#include "Inputs/cuda.h"

extern "C" __global__ void ckernel(int *a) {
  *a = 1;
}

// Kernel symbol for launching kernel.
// CHECK: @[[SYM:ckernel]] = constant ptr @__device_stub__ckernel, align 8

// Device side kernel names
// CHECK: @[[CKERN:[0-9]*]] = {{.*}} c"ckernel\00"

// DEV: define {{.*}}@ckernel{{.*}}!dbg
// DEV:  store {{.*}}!dbg
// DEV:  ret {{.*}}!dbg

// Make sure there is no !dbg between function attributes and '{'
// CHECK: define{{.*}} void @[[CSTUB:__device_stub__ckernel]]{{.*}} #{{[0-9]+}} {
// CHECK-NOT: call {{.*}}@hipLaunchByPtr{{.*}}!dbg
// CHECK: call {{.*}}@hipLaunchByPtr{{.*}}@[[SYM]]
// CHECK-NOT: ret {{.*}}!dbg

// CHECK-LABEL: define {{.*}}@_Z8hostfuncPi{{.*}}!dbg
// O0: call void @[[CSTUB]]{{.*}}!dbg
void hostfunc(int *a) {
  ckernel<<<1, 1>>>(a);
}
