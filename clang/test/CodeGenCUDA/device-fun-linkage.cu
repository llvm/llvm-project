// RUN: %clang_cc1 -triple nvptx -fcuda-is-device -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefix=NORDC %s
// RUN: %clang_cc1 -triple nvptx -fcuda-is-device -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefix=NORDC-NEG %s
// RUN: %clang_cc1 -triple nvptx -fcuda-is-device -fgpu-rdc -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefix=RDC %s
// RUN: %clang_cc1 -triple nvptx -fcuda-is-device -fgpu-rdc -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefix=RDC-NEG %s

#include "Inputs/cuda.h"

template <typename T> __device__ void func() {}
template <typename T> __global__ void kernel() {}

template __device__ void func<int>();
// NORDC:     define internal void @_Z4funcIiEvv()
// RDC:       define weak_odr void @_Z4funcIiEvv()

template __global__ void kernel<int>();
// NORDC:     define void @_Z6kernelIiEvv()
// RDC:       define weak_odr void @_Z6kernelIiEvv()

// Ensure that unused static device function is eliminated
static __device__ void static_func() {}
// NORDC-NEG-NOT: define{{.*}} void @_ZL13static_funcv()
// RDC-NEG-NOT:   define{{.*}} void @_ZL13static_funcv()

// Ensure that kernel function has external or weak_odr
// linkage regardless static specifier
static __global__ void static_kernel() {}
// NORDC:     define void @_ZL13static_kernelv()
// RDC:       define weak_odr void @_ZL13static_kernelv()
