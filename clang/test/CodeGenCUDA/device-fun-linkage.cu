// RUN: %clang_cc1 -triple nvptx -fcuda-is-device \
// RUN:   -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefix=NORDC %s
// RUN: %clang_cc1 -triple nvptx -fcuda-is-device \
// RUN:   -fgpu-rdc -emit-llvm -o - %s \
// RUN:   | FileCheck -check-prefix=RDC %s

#include "Inputs/cuda.h"

// NORDC: define internal void @_Z4funcIiEvv()
// NORDC: define void @_Z6kernelIiEvv()
// RDC: define weak_odr void @_Z4funcIiEvv()
// RDC: define weak_odr void @_Z6kernelIiEvv()

template <typename T> __device__ void func() {}
template <typename T> __global__ void kernel() {}

template __device__ void func<int>();
template __global__ void kernel<int>();
