// RUN: %clang_cc1 -fcxx-exceptions -fcuda-is-device -fsyntax-only -std=c++17 -verify %s

#include "Inputs/cuda.h"

// Error, instantiated on device.
inline __host__ __device__ void hasInvalid() {
  throw NULL;
  // expected-error@-1 2{{cannot use 'throw' in __host__ __device__ function}}
}

inline __host__ __device__ void hasInvalid2() {
  throw NULL;
  // expected-error@-1 2{{cannot use 'throw' in __host__ __device__ function}}
}

inline __host__ __device__ void hasInvalidDiscarded() {
  // This is only used in the discarded statements below, so this should not diagnose.
  throw NULL;
}

static __device__ void use0() {
  hasInvalid(); // expected-note {{called by 'use0'}}
  hasInvalid(); // expected-note {{called by 'use0'}}

  if constexpr (true) {
    hasInvalid2(); // expected-note {{called by 'use0'}}
  } else {
    hasInvalidDiscarded();
  }

  if constexpr (false) {
    hasInvalidDiscarded();
  } else {
    hasInvalid2(); // expected-note {{called by 'use0'}}
  }

  if constexpr (false) {
    hasInvalidDiscarded();
  }
}

// To avoid excessive diagnostic messages, deferred diagnostics are only
// emitted the first time a function is called.
static __device__ void use1() {
  use0(); // expected-note 4{{called by 'use1'}}
  use0();
}

static __device__ void use2() {
  use1(); // expected-note 4{{called by 'use2'}}
  use1();
}

static __device__ void use3() {
  use2(); // expected-note 4{{called by 'use3'}}
  use2();
}

__global__ void use4() {
  use3(); // expected-note 4{{called by 'use4'}}
  use3();
}
