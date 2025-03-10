// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -fcuda-is-device %s
// RUN: %clang_cc1 -fsyntax-only -verify -fcuda-is-device %s \
// RUN:   -fatomic-fine-grained-memory -fatomic-ignore-denormal-mode

#include "Inputs/cuda.h"

#if !__has_extension(clang_atomic_attributes)
#error "We should have atomic attributes support"
#endif

[[clang::atomic(!no_remote_memory)]] // expected-error {{use of undeclared identifier 'no_remote_memory'}}
__device__ __host__ void test_location(float *a) {
  __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  [[clang::atomic(!no_remote_memory)]] int x; // expected-error {{use of undeclared identifier 'no_remote_memory'}}
}

__device__ __host__ void test_invalid_option(float *a) {
  [[clang::atomic(fast)]] { // expected-error {{invalid argument 'fast' to atomic attribute; valid options are: 'remote_memory', 'fine_grained_memory', 'ignore_denormal_mode' (optionally prefixed with 'no_')}}
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __host__ void test_invalid_value(float *a) {
  [[clang::atomic(no_remote_memory(default))]] { // expected-error2 {{expected ','}} expected-error {{expected ')'}}
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __host__ void test_invalid_format(float *a) {
  [[clang::atomic(no_remote_memory=on)]] { // expected-error2 {{expected ','}} expected-error {{expected ')'}}
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

[[clang::atomic(no_remote_memory)]] // expected-error {{'atomic' attribute cannot be applied to a declaration}}
__device__ __host__ void test_not_compound_stmt(float *a) {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

__device__ __host__ void test_quoted(float *a) {
  [[clang::atomic("no_remote_memory", "remote_memory")]] { // expected-error {{'atomic' attribute requires an identifier}}
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __host__ void test_one_value(float *a) {
  [[clang::atomic(no_remote_memory)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __host__ void test_multiple_value(float *a) {
  [[clang::atomic(no_remote_memory, fine_grained_memory)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __host__ void test_duplicate_value(float *a) {
  [[clang::atomic(no_remote_memory, no_remote_memory)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __host__ void test_conflict_value(float *a) {
  [[clang::atomic(no_remote_memory, remote_memory)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

__device__ __host__ void test_multiple_attrs(float *a) {
  [[clang::atomic(no_remote_memory)]] [[clang::atomic(remote_memory)]] {
    __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}
