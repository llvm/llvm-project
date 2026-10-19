// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -fclangir \
// RUN:   -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -fclangir \
// RUN:   -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -fclangir \
// RUN:   -emit-llvm -x hip %s -o %t-hip-cir.ll
// RUN: FileCheck --input-file=%t-hip-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

#include "Inputs/cuda.h"

__device__ void f() {}

__global__ void kernel1(int a) { f(); }

// CIR: cir.func {{.*}} @_Z1fv(
// CIR-NOT: norecurse
// CIR: cir.func {{.*}} @_Z7kernel1i({{.*}} attributes {{.*}}norecurse

// LLVM: define {{.*}} @_Z1fv() #[[F_ATTR:[0-9]+]]
// LLVM: define {{.*}} @_Z7kernel1i({{.*}}) #[[K_ATTR:[0-9]+]]
// LLVM-NOT: attributes #[[F_ATTR]] = {{.*}}norecurse
// LLVM: attributes #[[K_ATTR]] = {{.*}}norecurse
