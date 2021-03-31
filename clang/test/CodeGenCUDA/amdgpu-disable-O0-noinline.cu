// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -emit-llvm -disable-O0-optnone -disable-O0-noinline -o - %s | FileCheck --check-prefix=CHECK %s

#include "Inputs/cuda.h"

// CHECK-NOT: Function Attrs: {{.*}} optnone
// removed  : Function Attrs: {{.*}} noinline

__device__ void foo() {
}

__global__ void bar() {
}
