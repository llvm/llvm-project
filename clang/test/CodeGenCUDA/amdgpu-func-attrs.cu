// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     | FileCheck -check-prefixes=NO-UNSAFE-FP-ATOMICS %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa \
// RUN:     -fcuda-is-device -emit-llvm -o - -x hip %s \
// RUN:     -munsafe-fp-atomics \
// RUN:     | FileCheck -check-prefixes=UNSAFE-FP-ATOMICS %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
// RUN:     -o - -x hip %s -munsafe-fp-atomics \
// RUN:     | FileCheck -check-prefix=NO-UNSAFE-FP-ATOMICS %s

#include "Inputs/cuda.h"

__device__ void test() {
// UNSAFE-FP-ATOMICS: define void @_Z4testv() [[ATTR:#[0-9]+]]
}


// Make sure this is silently accepted on other targets.
// NO-UNSAFE-FP-ATOMICS-NOT: "amdgpu-unsafe-fp-atomics"

// UNSAFE-FP-ATOMICS-DAG: attributes [[ATTR]] = {{.*}}"amdgpu-unsafe-fp-atomics"="true"
