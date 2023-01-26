// RUN: %clang_cc1 -triple x86_64-unknown-gnu-linux -aux-triple amdgcn-amd-amdhsa \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck -check-prefixes=CHECK,LNX %s
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -aux-triple amdgcn-amd-amdhsa \
// RUN:   -emit-llvm -o - -x hip %s | FileCheck -check-prefixes=CHECK,MSVC %s

#include "Inputs/cuda.h"

namespace X {
  __global__ void kern1(int *x);
  __device__ int var1;
}

// CHECK: @[[STR1:.*]] = {{.*}} c"_ZN1X5kern1EPi\00"
// CHECK: @[[STR2:.*]] = {{.*}} c"_ZN1X4var1E\00"

// LNX-LABEL: define {{.*}}@_Z4fun1v()
// MSVC-LABEL: define {{.*}} @"?fun1@@YAPEBDXZ"()
// CHECK: ret ptr @[[STR1]]
const char *fun1() {
  return __builtin_get_device_side_mangled_name(X::kern1);
}

// LNX-LABEL: define {{.*}}@_Z4fun2v()
// MSVC-LABEL: define {{.*}}@"?fun2@@YAPEBDXZ"()
// CHECK: ret ptr @[[STR2]]
__host__ __device__ const char *fun2() {
  return __builtin_get_device_side_mangled_name(X::var1);
}
