// RUN: %clang_cc1 -x hip -triple amdgcn -fcuda-is-device -verify -fsyntax-only -Whip-deprecated-builtins %s
// RUN: %clang_cc1 -x hip -triple amdgcn -fcuda-is-device -fdiagnostics-parseable-fixits -Whip-deprecated-builtins %s 2>&1 | FileCheck %s

#define __device__ __attribute__((device))

__device__ void test(int *p, int v, int s) {
  __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);     // expected-warning {{builtin __hip_atomic_load is deprecated; use __scoped_atomic_load_n instead}}
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); // expected-warning {{builtin __hip_atomic_store is deprecated; use __scoped_atomic_store_n instead}}
  __hip_atomic_load(p, __ATOMIC_RELAXED, s);                             // expected-warning {{builtin __hip_atomic_load is deprecated; use __scoped_atomic_load_n instead}}
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, s);                         // expected-warning {{builtin __hip_atomic_store is deprecated; use __scoped_atomic_store_n instead}}
}

// Statically known scope: both the builtin name and the scope argument get fix-its.
// CHECK: fix-it:{{.*}}:{[[@LINE-7]]:3-[[@LINE-7]]:20}:"__scoped_atomic_load_n"
// CHECK: fix-it:{{.*}}:{[[@LINE-8]]:42-[[@LINE-8]]:67}:"__MEMORY_SCOPE_SYSTEM"
// CHECK: fix-it:{{.*}}:{[[@LINE-8]]:3-[[@LINE-8]]:21}:"__scoped_atomic_store_n"
// CHECK: fix-it:{{.*}}:{[[@LINE-9]]:46-[[@LINE-9]]:71}:"__MEMORY_SCOPE_SYSTEM"
// Dynamic scope: only the builtin name gets a fix-it.
// CHECK: fix-it:{{.*}}:{[[@LINE-10]]:3-[[@LINE-10]]:20}:"__scoped_atomic_load_n"
// CHECK: fix-it:{{.*}}:{[[@LINE-10]]:3-[[@LINE-10]]:21}:"__scoped_atomic_store_n"

// Fetch ops: argument layout matches __scoped_atomic_fetch_*.  Scope varies
// across all six __HIP_MEMORY_SCOPE_* values.
__device__ void test_fetch(int *p, int v) {
  __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD); // expected-warning {{builtin __hip_atomic_fetch_add is deprecated; use __scoped_atomic_fetch_add instead}}
  __hip_atomic_fetch_sub(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);    // expected-warning {{builtin __hip_atomic_fetch_sub is deprecated; use __scoped_atomic_fetch_sub instead}}
  __hip_atomic_fetch_and(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);    // expected-warning {{builtin __hip_atomic_fetch_and is deprecated; use __scoped_atomic_fetch_and instead}}
  __hip_atomic_fetch_or(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);         // expected-warning {{builtin __hip_atomic_fetch_or is deprecated; use __scoped_atomic_fetch_or instead}}
  __hip_atomic_fetch_xor(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);       // expected-warning {{builtin __hip_atomic_fetch_xor is deprecated; use __scoped_atomic_fetch_xor instead}}
  __hip_atomic_fetch_min(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_CLUSTER);      // expected-warning {{builtin __hip_atomic_fetch_min is deprecated; use __scoped_atomic_fetch_min instead}}
  __hip_atomic_fetch_max(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SINGLETHREAD); // expected-warning {{builtin __hip_atomic_fetch_max is deprecated; use __scoped_atomic_fetch_max instead}}
}
// CHECK: fix-it:{{.*}}:{[[@LINE-8]]:3-[[@LINE-8]]:25}:"__scoped_atomic_fetch_add"
// CHECK: fix-it:{{.*}}:{[[@LINE-9]]:50-[[@LINE-9]]:81}:"__MEMORY_SCOPE_SINGLE"
// CHECK: fix-it:{{.*}}:{[[@LINE-9]]:3-[[@LINE-9]]:25}:"__scoped_atomic_fetch_sub"
// CHECK: fix-it:{{.*}}:{[[@LINE-10]]:50-[[@LINE-10]]:78}:"__MEMORY_SCOPE_WVFRNT"
// CHECK: fix-it:{{.*}}:{[[@LINE-10]]:3-[[@LINE-10]]:25}:"__scoped_atomic_fetch_and"
// CHECK: fix-it:{{.*}}:{[[@LINE-11]]:50-[[@LINE-11]]:78}:"__MEMORY_SCOPE_WRKGRP"
// CHECK: fix-it:{{.*}}:{[[@LINE-11]]:3-[[@LINE-11]]:24}:"__scoped_atomic_fetch_or"
// CHECK: fix-it:{{.*}}:{[[@LINE-12]]:49-[[@LINE-12]]:73}:"__MEMORY_SCOPE_DEVICE"
// CHECK: fix-it:{{.*}}:{[[@LINE-12]]:3-[[@LINE-12]]:25}:"__scoped_atomic_fetch_xor"
// CHECK: fix-it:{{.*}}:{[[@LINE-13]]:50-[[@LINE-13]]:75}:"__MEMORY_SCOPE_SYSTEM"
// CHECK: fix-it:{{.*}}:{[[@LINE-13]]:3-[[@LINE-13]]:25}:"__scoped_atomic_fetch_min"
// CHECK: fix-it:{{.*}}:{[[@LINE-14]]:50-[[@LINE-14]]:76}:"__MEMORY_SCOPE_CLUSTR"
// CHECK: fix-it:{{.*}}:{[[@LINE-14]]:3-[[@LINE-14]]:25}:"__scoped_atomic_fetch_max"
// CHECK: fix-it:{{.*}}:{[[@LINE-15]]:50-[[@LINE-15]]:81}:"__MEMORY_SCOPE_SINGLE"

// Exchange op: argument layout matches __scoped_atomic_exchange_n, so both
// name and scope get fix-its.  compare_exchange_{weak,strong} have a different
// argument layout (scoped form takes an additional 'weak' boolean and passes
// 'desired' by pointer), so only a deprecation warning is emitted.
__device__ void test_exchange(int *p, int *ep, int v) {
  __hip_atomic_exchange(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT);                               // expected-warning {{builtin __hip_atomic_exchange is deprecated; use __scoped_atomic_exchange_n instead}}
  __hip_atomic_compare_exchange_weak(p, ep, v, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);   // expected-warning {{builtin __hip_atomic_compare_exchange_weak is deprecated; use __scoped_atomic_compare_exchange instead}}
  __hip_atomic_compare_exchange_strong(p, ep, v, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);     // expected-warning {{builtin __hip_atomic_compare_exchange_strong is deprecated; use __scoped_atomic_compare_exchange instead}}
}
// CHECK: fix-it:{{.*}}:{[[@LINE-4]]:3-[[@LINE-4]]:24}:"__scoped_atomic_exchange_n"
// CHECK: fix-it:{{.*}}:{[[@LINE-5]]:49-[[@LINE-5]]:77}:"__MEMORY_SCOPE_WVFRNT"
