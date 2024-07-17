// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -fcuda-is-device %s
// RUN: %clang_cc1 -fsyntax-only -verify -fcuda-is-device %s \
// RUN:   -fatomic=no_fine_grained_memory:off,no_remote_memory:on,ignore_denormal_mode:on

#include "Inputs/cuda.h"

#pragma clang atomic no_remote_memory(off) // expected-error {{'#pragma clang atomic' can only appear at the start of a compound statement}}

__device__ __host__ void test_location(float *a) {
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  #pragma clang atomic no_remote_memory(off) // expected-error {{'#pragma clang atomic' can only appear at the start of a compound statement}}
}

__device__ __host__ void test_invalid_option(float *a) {
  #pragma clang atomic fast(on) // expected-error {{invalid option 'fast'; expected 'no_remote_memory', 'no_fine_grained_memory', or 'ignore_denormal_mode'}}
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

__device__ __host__ void test_invalid_value(float *a) {
  #pragma clang atomic no_remote_memory(default)  // expected-error {{unexpected argument 'default' to '#pragma clang atomic no_remote_memory'; expected 'on' or 'off'}}
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

__device__ __host__ void test_extra_token(float *a) {
  #pragma clang atomic no_remote_memory(on) * // expected-warning {{extra tokens at end of '#pragma clang atomic' - ignored}}
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}
