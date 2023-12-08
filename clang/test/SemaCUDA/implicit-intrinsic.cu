// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -fcuda-is-device \
// RUN:     -fsyntax-only -verify %s

#include "Inputs/cuda.h"

// expected-no-diagnostics
__device__ void __threadfence_system() {
  // This shouldn't produce an error, since __nvvm_membar_sys should be
  // __device__ and thus callable from device code.
  __nvvm_membar_sys();
}
