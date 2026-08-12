// RUN: %clang_cc1 %s -verify -fsyntax-only -triple=x86_64-unknown-linux-gnu -std=c11
// RUN: %clang_cc1 %s -verify -fsyntax-only -triple=amdgcn-amd-amdhsa -std=c11

typedef _Float16 half2 __attribute__((ext_vector_type(2)));
typedef __bf16 bfloat2 __attribute__((ext_vector_type(2)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef int int2 __attribute__((ext_vector_type(2)));
typedef unsigned int uint4 __attribute__((ext_vector_type(4)));
typedef _Bool bool8 __attribute__((ext_vector_type(8)));
typedef _BitInt(16) bitint2 __attribute__((ext_vector_type(2)));

void test_gnu(half2 *h2, bfloat2 *b2, float2 *f2, int2 *i2, uint4 *u4) {
  (void)__atomic_load_n(h2, __ATOMIC_RELAXED);
  __atomic_store_n(h2, *h2, __ATOMIC_RELAXED);
  (void)__atomic_exchange_n(h2, *h2, __ATOMIC_RELAXED);

  (void)__atomic_fetch_add(h2, *h2, __ATOMIC_RELAXED);
  (void)__atomic_fetch_add(b2, *b2, __ATOMIC_RELAXED);
  (void)__atomic_fetch_sub(f2, *f2, __ATOMIC_RELAXED);
  (void)__atomic_add_fetch(f2, *f2, __ATOMIC_RELAXED);
  (void)__atomic_fetch_min(f2, *f2, __ATOMIC_RELAXED);
  (void)__atomic_max_fetch(i2, *i2, __ATOMIC_RELAXED);
  (void)__atomic_fetch_fmaximum(f2, *f2, __ATOMIC_RELAXED);
  (void)__atomic_fetch_and(i2, *i2, __ATOMIC_RELAXED);
  (void)__atomic_or_fetch(u4, *u4, __ATOMIC_RELAXED);
  (void)__atomic_nand_fetch(i2, *i2, __ATOMIC_RELAXED);

  (void)__atomic_fetch_and(f2, *f2, __ATOMIC_RELAXED); // expected-error {{must be a pointer to integer}}
  (void)__atomic_fetch_fminimum(i2, *i2, __ATOMIC_RELAXED); // expected-error {{must be a pointer to floating point type}}
}

void test_c11(_Atomic(half2) *h2, half2 h2v, _Atomic(int2) *i2, int2 i2v) {
  (void)__c11_atomic_load(h2, __ATOMIC_RELAXED);
  __c11_atomic_store(h2, h2v, __ATOMIC_RELAXED);
  (void)__c11_atomic_exchange(h2, h2v, __ATOMIC_RELAXED);
  (void)__c11_atomic_fetch_add(h2, h2v, __ATOMIC_RELAXED);
  (void)__c11_atomic_fetch_sub(h2, h2v, __ATOMIC_RELAXED);
  (void)__c11_atomic_fetch_xor(i2, i2v, __ATOMIC_RELAXED);
}

void test_scoped(half2 *h2, int2 *i2) {
  (void)__scoped_atomic_load_n(h2, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  __scoped_atomic_store_n(h2, *h2, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  (void)__scoped_atomic_fetch_add(h2, *h2, __ATOMIC_RELAXED,
                                  __MEMORY_SCOPE_DEVICE);
  (void)__scoped_atomic_fetch_or(i2, *i2, __ATOMIC_RELAXED,
                                 __MEMORY_SCOPE_DEVICE);
}

void test_unsupported(float3 *f3, float8 *f8, bool8 *b8, bitint2 *bi2) {
  (void)__atomic_fetch_add(f3, *f3, __ATOMIC_RELAXED); // expected-error {{must be a pointer to a vector with a power-of-two size of at most 16 bytes}}
  (void)__atomic_fetch_add(f8, *f8, __ATOMIC_RELAXED); // expected-error {{must be a pointer to a vector with a power-of-two size of at most 16 bytes}}
  (void)__atomic_fetch_add(b8, *b8, __ATOMIC_RELAXED); // expected-error {{must be a pointer to integer, pointer or supported floating point type}}
  (void)__atomic_fetch_add(bi2, *bi2, __ATOMIC_RELAXED); // expected-error {{argument to atomic builtin of type '_BitInt' is not supported}}
}
