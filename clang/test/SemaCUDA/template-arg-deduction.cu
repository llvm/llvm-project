// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsyntax-only -fcuda-is-device -verify %s

// expected-no-diagnostics

#include "Inputs/cuda.h"

void foo();
__device__ void foo();

template<class F>
void host_temp(F f);

template<class F>
__device__ void device_temp(F f);

void host_caller() {
  host_temp(foo);
}

__global__ void kernel_caller() {
  device_temp(foo);
}

__device__ void device_caller() {
  device_temp(foo);
}
