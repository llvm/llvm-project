// RUN: echo "GPU binary would be here" > %t

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s \
// RUN:     -fcuda-include-gpubinary %t -o - -x hip\
// RUN:   | FileCheck -allow-deprecated-dag-overlap %s --check-prefixes=CHECK

#include "Inputs/cuda.h"

template<class T>
__global__ void kernelfunc() {}

// CHECK-LABEL: define{{.*}}@_Z8hostfuncv()
// CHECK: call void @[[STUB:__device_stub__Z10kernelfuncIiEvv]]()
void hostfunc(void) { kernelfunc<int><<<1, 1>>>(); }

// CHECK: define{{.*}}@[[STUB]]
// CHECK: call{{.*}}@hipLaunchByPtr{{.*}}@[[STUB]]

// CHECK-LABEL: define{{.*}}@__hip_register_globals
// CHECK: call{{.*}}@__hipRegisterFunction{{.*}}@[[STUB]]
