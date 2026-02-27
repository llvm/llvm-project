// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -o /dev/null -verify %s

#include <arm_acle.h>

void test_signed_ok(int *p, int v) {
  __builtin_arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 0);
}

void test_invalid_retention_policy(unsigned int *p, unsigned int v) {
  __builtin_arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 2);
  // expected-error@-1 {{argument value 2 is outside the valid range [0, 1]}}
}

void test_const_pointer(const unsigned int *p, unsigned int v) {
  __builtin_arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 0);
  // expected-error@-1 {{address argument to atomic builtin cannot be const-qualified}}
}

void test_non_integer_pointer(float *p, float v) {
  __builtin_arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 0);
  // expected-error@-1 {{address argument to '__arm_atomic_store_with_stshh' must be a pointer to an 8,16,32, or 64-bit integer type}}
}

void test_invalid_bit_width(__int128 *p, __int128 v) {
  __builtin_arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 0);
  // expected-error@-1 {{address argument to '__arm_atomic_store_with_stshh' must be a pointer to an 8,16,32, or 64-bit integer type}}
}

void test_invalid_memory_order(unsigned int *p, unsigned int v) {
  __builtin_arm_atomic_store_with_stshh(p, v, __ATOMIC_ACQUIRE, 0);
  // expected-error@-1 {{memory order argument to '__arm_atomic_store_with_stshh' must be one of __ATOMIC_RELAXED, __ATOMIC_RELEASE, or __ATOMIC_SEQ_CST}}
}

void test_invalid_memory_order_consume(unsigned int *p, unsigned int v) {
  __builtin_arm_atomic_store_with_stshh(p, v, __ATOMIC_CONSUME, 0);
  // expected-error@-1 {{memory order argument to '__arm_atomic_store_with_stshh' must be one of __ATOMIC_RELAXED, __ATOMIC_RELEASE, or __ATOMIC_SEQ_CST}}
}

void test_invalid_memory_order_acq_rel(unsigned int *p, unsigned int v) {
  __builtin_arm_atomic_store_with_stshh(p, v, __ATOMIC_ACQ_REL, 0);
  // expected-error@-1 {{memory order argument to '__arm_atomic_store_with_stshh' must be one of __ATOMIC_RELAXED, __ATOMIC_RELEASE, or __ATOMIC_SEQ_CST}}
}

void test_value_size_mismatch(int *p, short v) {
  __builtin_arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 0);
  // expected-error@-1 {{value argument to '__arm_atomic_store_with_stshh' must be 'int'; got 'short'}}
}

void test_non_integer_value(int *p, float v) {
  __builtin_arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 0);
  // expected-error@-1 {{value argument to '__arm_atomic_store_with_stshh' must be 'int'; got 'float'}}
}

void test_too_few_args(int *p, int v) {
  __builtin_arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED);
  // expected-error@-1 {{too few arguments to function call, expected 4, have 3}}
}

void test_too_many_args(int *p, int v) {
  __builtin_arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 0, 1);
  // expected-error@-1 {{too many arguments to function call, expected 4, have 5}}
}

void test_value_i128_mismatch(int *p, __int128 v) {
  __builtin_arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 0);
  // expected-error@-1 {{value argument to '__arm_atomic_store_with_stshh' must be 'int'; got '__int128'}}
}
