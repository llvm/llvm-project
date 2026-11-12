// RUN: %clang_cc1 %s --std=c++11 -triple nvptx -o - \
// RUN:   -verify -fcuda-is-device -fsyntax-only -verify-ignore-unexpected=note
// RUN: %clang_cc1 %s --std=c++11 -fgpu-rdc -triple nvptx -o - \
// RUN:   -verify=rdc -fcuda-is-device -fsyntax-only -verify-ignore-unexpected=note
// rdc-no-diagnostics

#include "Inputs/cuda.h"

__global__ void kernel1();
__global__ void kernel2() {
  kernel1<<<1,1>>>(); // expected-error {{kernel launch from __device__ or __global__ function requires relocatable device code (i.e. requires -fgpu-rdc)}}
}
