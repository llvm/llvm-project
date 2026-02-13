// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +pcdphint \
// RUN:   -D__ARM_FEATURE_PCDPHINT -fsyntax-only -verify %s

#include <arm_acle.h>

void test_const_pointer(const unsigned int *p, unsigned int v) {
  __arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 0);
  // expected-error@-1 {{address argument to atomic builtin cannot be const-qualified}}
}

void test_non_integer_pointer(float *p, float v) {
  __arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 0);
  // expected-error@-1 {{address argument to '__arm_atomic_store_with_stshh' must be a pointer to an 8,16,32, or 64-bit integer type}}
}

void test_invalid_bit_width(__int128 *p, __int128 v) {
  __arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 0);
  // expected-error@-1 {{address argument to '__arm_atomic_store_with_stshh' must be a pointer to an 8,16,32, or 64-bit integer type}}
}

void test_invalid_memory_order(unsigned int *p, unsigned int v) {
  __arm_atomic_store_with_stshh(p, v, __ATOMIC_ACQUIRE, 0);
  // expected-error@-1 {{memory order argument to '__arm_atomic_store_with_stshh' must be one of __ATOMIC_RELAXED, __ATOMIC_RELEASE, or __ATOMIC_SEQ_CST}}
}

void test_invalid_retention_policy(unsigned int *p, unsigned int v) {
  __arm_atomic_store_with_stshh(p, v, __ATOMIC_RELAXED, 2);
  // expected-error@-1 {{argument value 2 is outside the valid range [0, 1]}}
}
