// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -fsyntax-only \
// RUN:   -verify -Wno-vla %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -fsyntax-only \
// RUN:   -verify -Wno-vla %s

// NOTE: Do not autogenerate. Tests deferred diagnostic deduplication.

// Tests that deferred diagnostics are emitted once per function, with all
// callers listed as notes, rather than repeating the diagnostics for each
// caller. See https://github.com/llvm/llvm-project/issues/180638.

#include "Inputs/cuda.h"

__host__ void hf(); // expected-note 3{{'hf' declared here}}

// Lambda calling a host function. Its deferred diagnostics should be
// emitted only once even when multiple device functions call it.
__device__ auto l =
  [] {
    hf(); // expected-error {{reference to __host__ function 'hf' in __host__ __device__ function}}
    hf(); // expected-error {{reference to __host__ function 'hf' in __host__ __device__ function}}
  };

__device__ void df1() {
  l(); // expected-note {{called by 'df1'}}
}

__device__ void df2() {
  l(); // expected-note {{called by 'df2'}}
}

__device__ void df3() {
  l(); // expected-note {{called by 'df3'}}
}

// Test with shared call chains: two chains reaching the same function
// through different intermediate callers.
inline __host__ __device__ void hdf() {
  hf(); // expected-error {{reference to __host__ function 'hf' in __host__ __device__ function}}
}

inline __host__ __device__ void mid1() {
  hdf(); // expected-note {{called by 'mid1'}}
}

__device__ void dev1() {
  mid1(); // expected-note {{which is called by 'dev1'}}
}

inline __host__ __device__ void mid2() {
  hdf(); // expected-note {{called by 'mid2'}}
}

__device__ void dev2() {
  mid2(); // expected-note {{which is called by 'dev2'}}
}
