// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 \
// RUN:   -aux-triple x86_64-unknown-gnu-linux -fcuda-is-device \
// RUN:   -emit-llvm -o - -x hip %s 2>&1 | FileCheck %s

// RUN: %clang_cc1 -triple spirv64-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-gnu-linux -fcuda-is-device \
// RUN:   -emit-llvm -o - -x hip %s 2>&1 | FileCheck %s

// RUN: %clang_cc1 -triple nvptx \
// RUN:   -aux-triple x86_64-unknown-gnu-linux -fcuda-is-device \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck %s

// CHECK: @_ZN15infinity_helperIeE5valueE = {{.*}} double 0x47EFFFFFD586B834,{{.*}} align 8
// CHECK: @size = {{.*}} i32 8

#include "Inputs/cuda.h"

template <class> struct infinity_helper {};
template <> struct infinity_helper<long double> { static constexpr long double value = 3.4028234e38L; };
constexpr long double infinity_helper<long double>::value;
__device__ int size = sizeof(long double);
