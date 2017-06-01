// expected-no-diagnostics

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsyntax-only -fcuda-is-device -verify %s

#include "Inputs/cuda.h"

__host__ void overload() {}
__device__ void overload() {}

__host__ __device__ void test_hd() {
  // This should not be ambiguous -- we choose the host or the device overload
  // depending on whether or not we're compiling for host or device.
  void (*x)() = overload;
}

// These also shouldn't be ambiguous, but they're an easier test than the HD
// function above.
__host__ void test_host() {
  void (*x)() = overload;
}
__device__ void test_device() {
  void (*x)() = overload;
}
