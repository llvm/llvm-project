// RUN: %clang %s -O2 -o %t && %t
//===-- atomic_fp_minmax_test.c - Test FP atomic min/max operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests the floating-point atomic fetch-before min/max builtins,
// focusing on IEEE 754 corner cases: NaN, +/-infinity, +/-zero.
//
// Two families of operations:
// 1. fetch_fminimum/fetch_fmaximum: IEEE 754-2019 minimum/maximum
//    - Propagates NaN (any NaN input produces NaN output)
//    - Distinguishes -0 and +0 (minimum(-0, +0) = -0, maximum(-0, +0) = +0)
//
// 2. fetch_fminimum_num/fetch_fmaximum_num: IEEE 754-2019
// minimumNumber/maximumNumber
//    - Propagates numbers over NaN (minimumNumber(2.0, NaN) = 2.0)
//    - Treats -0 and +0 as equivalent
//
// All builtins return the old (pre-operation) value.
//
//===----------------------------------------------------------------------===//

#include <float.h>
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#undef NDEBUG
#include <assert.h>

// Memory order for all tests
#define MO memory_order_seq_cst

// Helper to check if a float is NaN
static inline bool is_nan_f(float x) { return x != x; }
static inline bool is_nan_d(double x) { return x != x; }

// Helper to check if two floats have the same bit pattern (for +0/-0
// distinction)
static inline bool same_bits_f(float a, float b) {
  uint32_t a_bits, b_bits;
  memcpy(&a_bits, &a, sizeof(float));
  memcpy(&b_bits, &b, sizeof(float));
  return a_bits == b_bits;
}

static inline bool same_bits_d(double a, double b) {
  uint64_t a_bits, b_bits;
  memcpy(&a_bits, &a, sizeof(double));
  memcpy(&b_bits, &b, sizeof(double));
  return a_bits == b_bits;
}

// Helper to create negative zero
static inline float neg_zero_f(void) { return -0.0f; }
static inline double neg_zero_d(void) { return -0.0; }

//===----------------------------------------------------------------------===//
// fetch_fminimum / fetch_fmaximum (propagates NaN, distinguishes zeros)
//===----------------------------------------------------------------------===//

void test_fetch_fminimum_float(void) {
  printf("Testing __atomic_fetch_fminimum (float)...\n");

  // Returns old value, stores minimum
  {
    float x = 5.0f;
    float old = __atomic_fetch_fminimum(&x, 3.0f, MO);
    assert(old == 5.0f && "fetch_fminimum should return old value");
    assert(x == 3.0f && "stored value should be 3.0");
  }

  {
    float x = 2.0f;
    float old = __atomic_fetch_fminimum(&x, 7.0f, MO);
    assert(old == 2.0f && "fetch_fminimum should return old value");
    assert(x == 2.0f && "stored value should be 2.0 (unchanged)");
  }

  // NaN propagation: fminimum(1.0, NaN) = NaN
  {
    float x = 1.0f;
    float old = __atomic_fetch_fminimum(&x, NAN, MO);
    assert(old == 1.0f && "fetch_fminimum should return old value");
    assert(is_nan_f(x) && "stored value should be NaN");
  }

  // NaN propagation: fminimum(NaN, 1.0) = NaN
  {
    float x = NAN;
    float old = __atomic_fetch_fminimum(&x, 1.0f, MO);
    assert(is_nan_f(old) && "fetch_fminimum should return old NaN");
    assert(is_nan_f(x) && "stored value should be NaN");
  }

  // Zero distinction: fminimum(+0, -0) = -0
  {
    float x = 0.0f;
    float old = __atomic_fetch_fminimum(&x, neg_zero_f(), MO);
    assert(same_bits_f(old, 0.0f) && "fetch_fminimum should return old +0");
    assert(same_bits_f(x, neg_zero_f()) && "stored value should be -0");
  }

  // Zero distinction: fminimum(-0, +0) = -0
  {
    float x = neg_zero_f();
    float old = __atomic_fetch_fminimum(&x, 0.0f, MO);
    assert(same_bits_f(old, neg_zero_f()) &&
           "fetch_fminimum should return old -0");
    assert(same_bits_f(x, neg_zero_f()) && "stored value should be -0");
  }

  // Infinity
  {
    float x = INFINITY;
    float old = __atomic_fetch_fminimum(&x, 1.0f, MO);
    assert(old == INFINITY && "fetch_fminimum should return old +inf");
    assert(x == 1.0f && "stored value should be 1.0");
  }

  {
    float x = -INFINITY;
    float old = __atomic_fetch_fminimum(&x, 1.0f, MO);
    assert(old == -INFINITY && "fetch_fminimum should return old -inf");
    assert(x == -INFINITY && "stored value should be -inf (unchanged)");
  }

  printf("  PASSED\n");
}

void test_fetch_fmaximum_float(void) {
  printf("Testing __atomic_fetch_fmaximum (float)...\n");

  // Returns old value, stores maximum
  {
    float x = 5.0f;
    float old = __atomic_fetch_fmaximum(&x, 3.0f, MO);
    assert(old == 5.0f && "fetch_fmaximum should return old value");
    assert(x == 5.0f && "stored value should be 5.0 (unchanged)");
  }

  {
    float x = 2.0f;
    float old = __atomic_fetch_fmaximum(&x, 7.0f, MO);
    assert(old == 2.0f && "fetch_fmaximum should return old value");
    assert(x == 7.0f && "stored value should be 7.0");
  }

  // NaN propagation: fmaximum(1.0, NaN) = NaN
  {
    float x = 1.0f;
    float old = __atomic_fetch_fmaximum(&x, NAN, MO);
    assert(old == 1.0f && "fetch_fmaximum should return old value");
    assert(is_nan_f(x) && "stored value should be NaN");
  }

  // Zero distinction: fmaximum(+0, -0) = +0
  {
    float x = 0.0f;
    float old = __atomic_fetch_fmaximum(&x, neg_zero_f(), MO);
    assert(same_bits_f(old, 0.0f) && "fetch_fmaximum should return old +0");
    assert(same_bits_f(x, 0.0f) && "stored value should be +0");
  }

  // Zero distinction: fmaximum(-0, +0) = +0
  {
    float x = neg_zero_f();
    float old = __atomic_fetch_fmaximum(&x, 0.0f, MO);
    assert(same_bits_f(old, neg_zero_f()) &&
           "fetch_fmaximum should return old -0");
    assert(same_bits_f(x, 0.0f) && "stored value should be +0");
  }

  // Infinity
  {
    float x = INFINITY;
    float old = __atomic_fetch_fmaximum(&x, 1.0f, MO);
    assert(old == INFINITY && "fetch_fmaximum should return old +inf");
    assert(x == INFINITY && "stored value should be +inf (unchanged)");
  }

  printf("  PASSED\n");
}

//===----------------------------------------------------------------------===//
// fetch_fminimum_num / fetch_fmaximum_num (propagates numbers, zeros equal)
//===----------------------------------------------------------------------===//

void test_fetch_fminimum_num_float(void) {
  printf("Testing __atomic_fetch_fminimum_num (float)...\n");

  // Returns old value, stores minimumnum
  {
    float x = 5.0f;
    float old = __atomic_fetch_fminimum_num(&x, 3.0f, MO);
    assert(old == 5.0f && "fetch_fminimum_num should return old value");
    assert(x == 3.0f && "stored value should be 3.0");
  }

  // Number over NaN: fminimumnum(1.0, NaN) = 1.0
  {
    float x = 1.0f;
    float old = __atomic_fetch_fminimum_num(&x, NAN, MO);
    assert(old == 1.0f && "fetch_fminimum_num should return old value");
    assert(x == 1.0f && "stored value should be 1.0 (number over NaN)");
  }

  // Number over NaN: fminimumnum(NaN, 2.0) = 2.0
  {
    float x = NAN;
    float old = __atomic_fetch_fminimum_num(&x, 2.0f, MO);
    assert(is_nan_f(old) && "fetch_fminimum_num should return old NaN");
    assert(x == 2.0f && "stored value should be 2.0 (number over NaN)");
  }

  // NaN + NaN = NaN
  {
    float x = NAN;
    float old = __atomic_fetch_fminimum_num(&x, NAN, MO);
    assert(is_nan_f(old) && "fetch_fminimum_num should return old NaN");
    assert(is_nan_f(x) && "stored value should be NaN");
  }

  // Infinity
  {
    float x = INFINITY;
    float old = __atomic_fetch_fminimum_num(&x, 1.0f, MO);
    assert(old == INFINITY && "fetch_fminimum_num should return old +inf");
    assert(x == 1.0f && "stored value should be 1.0");
  }

  printf("  PASSED\n");
}

void test_fetch_fmaximum_num_float(void) {
  printf("Testing __atomic_fetch_fmaximum_num (float)...\n");

  // Returns old value, stores maximumnum
  {
    float x = 5.0f;
    float old = __atomic_fetch_fmaximum_num(&x, 3.0f, MO);
    assert(old == 5.0f && "fetch_fmaximum_num should return old value");
    assert(x == 5.0f && "stored value should be 5.0 (unchanged)");
  }

  // Number over NaN: fmaximumnum(1.0, NaN) = 1.0
  {
    float x = 1.0f;
    float old = __atomic_fetch_fmaximum_num(&x, NAN, MO);
    assert(old == 1.0f && "fetch_fmaximum_num should return old value");
    assert(x == 1.0f && "stored value should be 1.0 (number over NaN)");
  }

  // Number over NaN: fmaximumnum(NaN, 2.0) = 2.0
  {
    float x = NAN;
    float old = __atomic_fetch_fmaximum_num(&x, 2.0f, MO);
    assert(is_nan_f(old) && "fetch_fmaximum_num should return old NaN");
    assert(x == 2.0f && "stored value should be 2.0 (number over NaN)");
  }

  printf("  PASSED\n");
}

//===----------------------------------------------------------------------===//
// Double precision tests
//===----------------------------------------------------------------------===//

void test_fetch_fminimum_double(void) {
  printf("Testing __atomic_fetch_fminimum (double)...\n");

  // NaN propagation
  {
    double x = 1.0;
    double old = __atomic_fetch_fminimum(&x, NAN, MO);
    assert(old == 1.0 && "fetch_fminimum should return old value");
    assert(is_nan_d(x) && "stored value should be NaN");
  }

  // Zero distinction: fminimum(+0, -0) = -0
  {
    double x = 0.0;
    double old = __atomic_fetch_fminimum(&x, neg_zero_d(), MO);
    assert(same_bits_d(old, 0.0) && "fetch_fminimum should return old +0");
    assert(same_bits_d(x, neg_zero_d()) && "stored value should be -0");
  }

  // Normal values
  {
    double x = 3.14;
    double old = __atomic_fetch_fminimum(&x, 2.71, MO);
    assert(old == 3.14 && "fetch_fminimum should return old value");
    assert(x == 2.71 && "stored value should be 2.71");
  }

  printf("  PASSED\n");
}

void test_fetch_fmaximum_double(void) {
  printf("Testing __atomic_fetch_fmaximum (double)...\n");

  // NaN propagation
  {
    double x = 1.0;
    double old = __atomic_fetch_fmaximum(&x, NAN, MO);
    assert(old == 1.0 && "fetch_fmaximum should return old value");
    assert(is_nan_d(x) && "stored value should be NaN");
  }

  // Zero distinction: fmaximum(-0, +0) = +0
  {
    double x = neg_zero_d();
    double old = __atomic_fetch_fmaximum(&x, 0.0, MO);
    assert(same_bits_d(old, neg_zero_d()) &&
           "fetch_fmaximum should return old -0");
    assert(same_bits_d(x, 0.0) && "stored value should be +0");
  }

  printf("  PASSED\n");
}

void test_fetch_fminimum_num_double(void) {
  printf("Testing __atomic_fetch_fminimum_num (double)...\n");

  // Number over NaN
  {
    double x = NAN;
    double old = __atomic_fetch_fminimum_num(&x, 2.5, MO);
    assert(is_nan_d(old) && "fetch_fminimum_num should return old NaN");
    assert(x == 2.5 && "stored value should be 2.5");
  }

  // Normal values
  {
    double x = 10.5;
    double old = __atomic_fetch_fminimum_num(&x, 8.3, MO);
    assert(old == 10.5 && "fetch_fminimum_num should return old value");
    assert(x == 8.3 && "stored value should be 8.3");
  }

  printf("  PASSED\n");
}

void test_fetch_fmaximum_num_double(void) {
  printf("Testing __atomic_fetch_fmaximum_num (double)...\n");

  // Number over NaN
  {
    double x = NAN;
    double old = __atomic_fetch_fmaximum_num(&x, 2.5, MO);
    assert(is_nan_d(old) && "fetch_fmaximum_num should return old NaN");
    assert(x == 2.5 && "stored value should be 2.5");
  }

  // Normal values
  {
    double x = 1.0;
    double old = __atomic_fetch_fmaximum_num(&x, 5.5, MO);
    assert(old == 1.0 && "fetch_fmaximum_num should return old value");
    assert(x == 5.5 && "stored value should be 5.5");
  }

  printf("  PASSED\n");
}

//===----------------------------------------------------------------------===//
// Main test runner
//===----------------------------------------------------------------------===//

int main(void) {
  printf("\n");
  printf("=============================================================\n");
  printf("Atomic Floating-Point Min/Max Tests\n");
  printf("=============================================================\n");
  printf("\n");

  printf("--- fetch_fminimum/fetch_fmaximum (propagate NaN, distinguish zeros) "
         "---\n");
  test_fetch_fminimum_float();
  test_fetch_fmaximum_float();
  test_fetch_fminimum_double();
  test_fetch_fmaximum_double();

  printf("\n--- fetch_fminimum_num/fetch_fmaximum_num (prefer numbers, treat "
         "zeros equal) ---\n");
  test_fetch_fminimum_num_float();
  test_fetch_fmaximum_num_float();
  test_fetch_fminimum_num_double();
  test_fetch_fmaximum_num_double();

  printf("\n");
  printf("=============================================================\n");
  printf("All tests PASSED!\n");
  printf("=============================================================\n");
  printf("\n");

  return 0;
}
