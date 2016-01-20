// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -fsyntax-only \
// RUN:   -verify -DEXPECT_ERR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only %s

#include <stdarg.h>
#include "Inputs/cuda.h"

__device__ void foo() {
  va_list list;
  va_arg(list, int);
#ifdef EXPECT_ERR
  // expected-error@-2 {{CUDA device code does not support va_arg}}
#endif
}

void bar() {
  va_list list;
  va_arg(list, int);  // OK: host-only
}

__device__ void baz() {
#if !defined(__CUDA_ARCH__)
  va_list list;
  va_arg(list, int);  // OK: only seen when compiling for host
#endif
}
