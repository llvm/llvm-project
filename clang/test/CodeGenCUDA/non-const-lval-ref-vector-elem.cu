// RUN: %clang_cc1 -emit-llvm %s -o - -fcuda-is-device -triple nvptx-unknown-unknown | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - -fcuda-is-device -triple amdgcn | FileCheck %s

#include "Inputs/cuda.h"
typedef double __attribute__((vector_size(32))) native_double4;

struct alignas(32) double4_struct {
    double x,y,z,w;
    __device__ native_double4& data() { return (native_double4&)(*this); }
};

// CHECK-LABEL: test_write
// CHECK:  %[[LD:.*]] = load <4 x double>
// CHECK:  %vecins = insertelement <4 x double> %[[LD]], double 1.000000e+00
// CHECK:  store <4 x double> %vecins
__device__ void test_write(double4_struct& x, int i) {
  x.data()[i] = 1;
}

// CHECK-LABEL: test_read
// CHECK:  %[[LD:.*]] = load <4 x double>
// CHECK:  %vecext = extractelement <4 x double> %[[LD]]
__device__ void test_read(double& y, double4_struct& x, int i) {
  y = x.data()[i];
}
