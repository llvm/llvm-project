// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -fsyntax-only \
// RUN:   -verify %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -fsyntax-only \
// RUN:   -verify %s

// NOTE: Do not autogenerate. Tests that "called by" notes are attached to
// all deferred diagnostics, not just the last one in a function.
// See https://github.com/llvm/llvm-project/issues/180638.

#include "Inputs/cuda.h"

__host__ void hf(); // expected-note 2{{'hf' declared here}}

__device__ auto l =
  [] { // expected-note 2{{in HD-promoted function 'operator()'}}
    hf(); // expected-error {{reference to __host__ function 'hf' in __host__ __device__ function}}
    hf(); // expected-error {{reference to __host__ function 'hf' in __host__ __device__ function}}
  };

__device__ void df1() {
  l(); // expected-note 2{{called by 'df1'}}
}

__device__ void df2() {
  l(); // expected-note 2{{called by 'df2'}}
}
