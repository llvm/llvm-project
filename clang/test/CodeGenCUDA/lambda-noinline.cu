// RUN: %clang_cc1 -x hip -emit-llvm -std=c++11 %s -o - \
// RUN:   -triple x86_64-linux-gnu \
// RUN:   | FileCheck -check-prefix=HOST %s
// RUN: %clang_cc1 -x hip -emit-llvm -std=c++11 %s -o - \
// RUN:   -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   | FileCheck -check-prefix=DEV %s

#include "Inputs/cuda.h"

// Checks noinline is correctly added to the lambda function.

// HOST: define{{.*}}@_ZZ4HostvENKUlvE_clEv({{.*}}) #[[ATTR:[0-9]+]]
// HOST: attributes #[[ATTR]]{{.*}}noinline

// DEV: define{{.*}}@_ZZ6DevicevENKUlvE_clEv({{.*}}) #[[ATTR:[0-9]+]]
// DEV: attributes #[[ATTR]]{{.*}}noinline

__device__ int a;
int b;

__device__ int Device() { return ([&] __device__ __noinline__ (){ return a; })(); }

__host__ int Host() { return ([&] __host__ __noinline__ (){ return b; })(); }
