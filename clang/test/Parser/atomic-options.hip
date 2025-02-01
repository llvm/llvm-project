// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -fcuda-is-device %s
// RUN: %clang_cc1 -fsyntax-only -verify -fcuda-is-device %s \
// RUN:   -fatomic=no_fine_grained_memory:off,no_remote_memory:on,ignore_denormal_mode:on

#include "Inputs/cuda.h"

#if !__has_extension(atomic_attributes)
#error "We should have atomic attributes support"
#endif

[[clang::atomic(!no_remote_memory)]] // expected-error {{'atomic' attribute cannot be applied to a declaration}}
__device__ __host__ void test_location(float *a) {
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  [[clang::atomic(!no_remote_memory)]] int x; // expected-error {{'atomic' attribute cannot be applied to a declaration}}
}

__device__ __host__ void test_invalid_option(float *a) {
  [[clang::atomic(fast)]] { // expected-error {{invalid argument 'fast' to atomic attribute; valid options are: 'no_remote_memory', 'no_fine_grained_memory', 'ignore_denormal_mode' (optionally prefixed with '!')}}
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __host__ void test_invalid_value(float *a) {
  [[clang::atomic(no_remote_memory(default))]] { // expected-error {{expected ')'}} expected-note {{to match this '('}}
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __host__ void test_invalid_format(float *a) {
  [[clang::atomic(no_remote_memory=on)]] { // expected-error {{expected ')'}} expected-note {{to match this '('}}
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}
