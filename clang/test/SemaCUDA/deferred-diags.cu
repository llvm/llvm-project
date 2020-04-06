// RUN: %clang_cc1 -fcxx-exceptions -fcuda-is-device -fsyntax-only -verify %s

#include "Inputs/cuda.h"

// Error, instantiated on device.
inline __host__ __device__ void hasInvalid() {
  throw NULL;
  // expected-error@-1 2{{cannot use 'throw' in __host__ __device__ function}}
}

static __device__ void use0() {
  hasInvalid(); // expected-note {{called by 'use0'}}
  hasInvalid(); // expected-note {{called by 'use0'}}
}

// To avoid excessive diagnostic messages, deferred diagnostics are only
// emitted the first time a function is called.
static __device__ void use1() {
  use0(); // expected-note 2{{called by 'use1'}}
  use0();
}

static __device__ void use2() {
  use1(); // expected-note 2{{called by 'use2'}}
  use1();
}

static __device__ void use3() {
  use2(); // expected-note 2{{called by 'use3'}}
  use2();
}

__global__ void use4() {
  use3(); // expected-note 2{{called by 'use4'}}
  use3();
}
