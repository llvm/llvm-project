// RUN: %clang_cc1 -fcxx-exceptions -fcuda-is-device -fsyntax-only -verify=enabled,device,device-enabled %s
// RUN: %clang_cc1 -fcxx-exceptions -fsyntax-only -verify=enabled %s
// RUN: %clang_cc1 -fcuda-is-device -fsyntax-only -verify=disabled,device,device-disabled %s
// RUN: %clang_cc1 -fsyntax-only -verify=disabled,host-disabled %s

#include "Inputs/cuda.h"

void host() {
  throw NULL; // host-disabled-error {{cannot use 'throw' with exceptions disabled}}
  try {} catch(void*) {} // host-disabled-error {{cannot use 'try' with exceptions disabled}}
}
__device__ void device() {
  throw NULL;
  // enabled-error@-1 {{cannot use 'throw' in __device__ function}} device-disabled-error@-1 {{cannot use 'throw' with exceptions disabled}}
  try {} catch(void*) {}
  // enabled-error@-1 {{cannot use 'try' in __device__ function}} device-disabled-error@-1 {{cannot use 'try' with exceptions disabled}}
}
__global__ void kernel() {
  throw NULL;
  // enabled-error@-1 {{cannot use 'throw' in __global__ function}} device-disabled-error@-1 {{cannot use 'throw' with exceptions disabled}}
  try {} catch(void*) {}
  // enabled-error@-1 {{cannot use 'try' in __global__ function}} device-disabled-error@-1 {{cannot use 'try' with exceptions disabled}}
}

// Check that it's an error to use 'try' and 'throw' from a __host__ __device__
// function if and only if it's codegen'ed for device.

__host__ __device__ void hd1() {
  throw NULL;
  // device-enabled-error@-1 {{cannot use 'throw' in __host__ __device__ function}} disabled-error@-1 {{cannot use 'throw' with exceptions disabled}}
  try {} catch(void*) {}
  // device-enabled-error@-1 {{cannot use 'try' in __host__ __device__ function}} disabled-error@-1 {{cannot use 'try' with exceptions disabled}}
}

// Error only on host; never instantiated on device.
inline __host__ __device__ void hd2() {
  throw NULL; // host-disabled-error {{cannot use 'throw' with exceptions disabled}}
  try {} catch(void*) {} // host-disabled-error {{cannot use 'try' with exceptions disabled}}
}
void call_hd2() { hd2(); } // host-disabled-note {{called by}}

// Error, instantiated on device.
inline __host__ __device__ void hd3() {
  throw NULL;
  // device-enabled-error@-1 {{cannot use 'throw' in __host__ __device__ function}} device-disabled-error@-1 {{cannot use 'throw' with exceptions disabled}}
  try {} catch(void*) {}
  // device-enabled-error@-1 {{cannot use 'try' in __host__ __device__ function}} device-disabled-error@-1 {{cannot use 'try' with exceptions disabled}}
}

__device__ void call_hd3() { hd3(); } // device-note {{called by}}

// Exceptions in uninstantiated/discarded code are not diagnosed.
template <typename>
__device__ void dev_not_instantiated() {
  throw 1;
  try {} catch(...) {}
}

template <typename>
__device__ void dev_instantiated() {
  throw 1;
  // enabled-error@-1 {{cannot use 'throw' in __device__ function}} device-disabled-error@-1 {{cannot use 'throw' with exceptions disabled}}
  try {} catch(...) {}
  // enabled-error@-1 {{cannot use 'try' in __device__ function}} device-disabled-error@-1 {{cannot use 'try' with exceptions disabled}}
}

template <bool x>
__device__ void dev_discarded() {
  if constexpr (x) {
    throw 1;
    try {} catch(...) {}
  }
}

__device__ void call_dev() {
  dev_instantiated<bool>(); // enabled-note {{in instantiation of}} device-disabled-note {{in instantiation of}}
  dev_discarded<false>();
}

template <typename>
__host__ __device__ void hd_not_instantiated() {
  throw 1;
  try {} catch(...) {}
}

template <typename>
__host__ __device__ void hd_instantiated() {
  throw 1;
  // device-enabled-error@-1 {{cannot use 'throw' in __host__ __device__ function}} disabled-error@-1 {{cannot use 'throw' with exceptions disabled}}
  try {} catch(...) {}
  // device-enabled-error@-1 {{cannot use 'try' in __host__ __device__ function}} disabled-error@-1 {{cannot use 'try' with exceptions disabled}}
}

template <bool x>
__host__ __device__ void hd_discarded() {
  if constexpr (x) {
    throw 1;
    try {} catch(...) {}
  }
}

__host__ __device__ void call_hd() {
  hd_instantiated<bool>(); // device-enabled-note {{called by}} disabled-note {{called by}}
  hd_discarded<false>();
}
