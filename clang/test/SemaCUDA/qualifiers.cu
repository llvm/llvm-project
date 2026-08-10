// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple nvptx-unknown-cuda -fsyntax-only -verify -fcuda-is-device %s
//
// We run clang_cc1 with 'not' because source file contains
// intentional errors. CC1 failure is expected and must be ignored
// here. We're interested in what ends up in AST and that's what
// FileCheck verifies.
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -ast-dump %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-ALL --check-prefix=CHECK-HOST
// RUN: not %clang_cc1 -triple nvptx-unknown-cuda -ast-dump -fcuda-is-device %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-ALL --check-prefix=CHECK-DEVICE

#include "Inputs/cuda.h"

// Host (x86) supports TLS and device-side compilation should ignore
// host variables. No errors in either case.
int __thread host_tls_var;
// CHECK-ALL: host_tls_var 'int' tls

#if defined(__CUDA_ARCH__)
// NVPTX does not support TLS
__device__ int __thread device_tls_var; // expected-error {{thread-local storage is not supported for the current target}}
// CHECK-DEVICE: device_tls_var 'int' tls
__shared__ int __thread shared_tls_var; // expected-error {{thread-local storage is not supported for the current target}}
// CHECK-DEVICE: shared_tls_var 'int' tls
#else
// Device-side vars should not produce any errors during host-side
// compilation.
__device__ int __thread device_tls_var;
// CHECK-HOST: device_tls_var 'int' tls
__shared__ int __thread shared_tls_var;
// CHECK-HOST: shared_tls_var 'int' tls
#endif

__global__ void g1(int x) {}
__global__ int g2(int x) { // expected-error {{must have void return type}}
  return 1;
}
