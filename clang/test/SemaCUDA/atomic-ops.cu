// RUN: %clang_cc1 -x hip -std=c++11 -triple amdgcn -fcuda-is-device -verify -fsyntax-only %s

#include "Inputs/cuda.h"

__device__ int test_hip_atomic_load(int *pi32, unsigned int *pu32, long long *pll, unsigned long long *pull, float *fp, double *dbl) {
  int val = __hip_atomic_load(0);      // expected-error {{too few arguments to function call, expected 3, have 1}}
  val = __hip_atomic_load(0, 0, 0, 0); // expected-error {{too many arguments to function call, expected 3, have 4}}
  val = __hip_atomic_load(0, 0, 0);    // expected-error {{address argument to atomic builtin must be a pointer ('int' invalid)}}
  val = __hip_atomic_load(pi32, 0, 0); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  val = __hip_atomic_load(pi32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_load(pi32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  val = __hip_atomic_load(pi32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  val = __hip_atomic_load(pi32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  val = __hip_atomic_load(pi32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  val = __hip_atomic_load(pi32, __ATOMIC_RELAXED, 6); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  val = __hip_atomic_load(pi32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_load(pi32, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_load(pi32, __ATOMIC_CONSUME, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_load(pi32, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_load(pi32, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SINGLETHREAD); // expected-warning{{memory order argument to atomic operation is invalid}}
  val = __hip_atomic_load(pu32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_load(pll, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_load(pull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_load(fp, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  val = __hip_atomic_load(dbl, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  return val;
}

__device__ int test_hip_atomic_store(int *pi32, unsigned int *pu32, long long *pll, unsigned long long *pull, float *fp, double *dbl,
                                     int i32, unsigned int u32, long long i64, unsigned long long u64, float f32, double f64) {
  __hip_atomic_store(0);             // expected-error {{too few arguments to function call, expected 4, have 1}}
  __hip_atomic_store(0, 0, 0, 0, 0); // expected-error {{too many arguments to function call, expected 4, have 5}}
  __hip_atomic_store(0, 0, 0, 0);    // expected-error {{address argument to atomic builtin must be a pointer ('int' invalid)}}
  __hip_atomic_store(pi32, 0, 0, 0); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  __hip_atomic_store(pi32, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(pi32, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  __hip_atomic_store(pi32, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  __hip_atomic_store(pi32, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_store(pi32, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(pi32, 0, __ATOMIC_RELAXED, 6); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  __hip_atomic_store(pi32, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(pi32, 0, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(pi32, 0, __ATOMIC_CONSUME, __HIP_MEMORY_SCOPE_SINGLETHREAD); // expected-warning{{memory order argument to atomic operation is invalid}}
  __hip_atomic_store(pi32, 0, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SINGLETHREAD); // expected-warning{{memory order argument to atomic operation is invalid}}
  __hip_atomic_store(pi32, 0, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SINGLETHREAD); // expected-warning{{memory order argument to atomic operation is invalid}}
  __hip_atomic_store(pi32, i32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(pi32, i32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(pu32, u32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(pll, i64, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(pull, u64, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(fp, f32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(dbl, f64, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(pi32, u32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(pi32, i64, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(pi32, u64, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(pll, i32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(fp, i32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(fp, i64, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(dbl, i64, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  __hip_atomic_store(dbl, i32, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  return 0;
}

__device__ bool test_hip_atomic_cmpxchg_weak(int *ptr, int val, int desired) {
  bool flag = __hip_atomic_compare_exchange_weak(0);                                     // expected-error {{too few arguments to function call, expected 6, have 1}}
  flag = __hip_atomic_compare_exchange_weak(0, 0, 0, 0, 0, 0, 0);                        // expected-error {{too many arguments to function call, expected 6, have 7}}
  flag = __hip_atomic_compare_exchange_weak(0, 0, 0, 0, 0, 0);                           // expected-error {{address argument to atomic builtin must be a pointer ('int' invalid)}}
  flag = __hip_atomic_compare_exchange_weak(ptr, 0, 0, 0, 0, 0);                         // expected-error {{synchronization scope argument to atomic operation is invalid}}, expected-warning {{null passed to a callee that requires a non-null argument}}
  flag = __hip_atomic_compare_exchange_weak(ptr, 0, 0, 0, 0, __HIP_MEMORY_SCOPE_SYSTEM); // expected-warning {{null passed to a callee that requires a non-null argument}}
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_CONSUME, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_CONSUME, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_CONSUME, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  flag = __hip_atomic_compare_exchange_weak(ptr, &val, desired, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD);
  return flag;
}
