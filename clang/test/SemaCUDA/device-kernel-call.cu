// RUN: %clang_cc1 -fcuda-is-device -verify=nordc %s
// RUN: %clang_cc1 -fcuda-is-device -fgpu-rdc -verify=rdc %s
// RUN: %clang_cc1 -x hip -fcuda-is-device -verify=hip %s

// rdc-no-diagnostics

#include "Inputs/cuda.h"

__global__ void g2(int x) {}

__global__ void g1(void) {
  g2<<<1, 1>>>(42);
  // nordc-error@-1 {{kernel launch from __device__ or __global__ function requires relocatable device code (i.e. requires -fgpu-rdc)}}
  // hip-error@-2 {{device-side kernel call/launch is not supported}}
}
