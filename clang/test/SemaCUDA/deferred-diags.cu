// RUN: %clang_cc1 -fcxx-exceptions -fcuda-is-device -fsyntax-only -std=c++17 -verify %s

#include "Inputs/cuda.h"

// Error, instantiated on device.
inline __host__ __device__ void hasInvalid() {
  throw NULL;
  // expected-error@-1 {{cannot use 'throw' in __host__ __device__ function}}
}

inline __host__ __device__ void hasInvalid2() {
  throw NULL;
  // expected-error@-1 {{cannot use 'throw' in __host__ __device__ function}}
}

inline __host__ __device__ void hasInvalidDiscarded() {
  // This is only used in the discarded statements below, so this should not diagnose.
  throw NULL;
}

static __device__ void use0() {
  hasInvalid(); // expected-note {{called by 'use0'}}
  hasInvalid();

  if constexpr (true) {
    hasInvalid2(); // expected-note {{called by 'use0'}}
  } else {
    hasInvalidDiscarded();
  }

  if constexpr (false) {
    hasInvalidDiscarded();
  } else {
    hasInvalid2();
  }

  if constexpr (false) {
    hasInvalidDiscarded();
  }
}

// Deferred diagnostics are emitted once per function, with all callers
// listed as notes.
static __device__ void use1() {
  use0(); // expected-note 2{{which is called by 'use1'}}
  use0();
}

static __device__ void use2() {
  use1(); // expected-note 2{{which is called by 'use2'}}
  use1();
}

static __device__ void use3() {
  use2(); // expected-note 2{{which is called by 'use3'}}
  use2();
}

__global__ void use4() {
  use3(); // expected-note 2{{which is called by 'use4'}}
  use3();
}
