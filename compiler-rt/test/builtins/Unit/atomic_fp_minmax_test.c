// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: native-run
//===-- atomic_fp_minmax_test.c - Test FP atomic min/max operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file tests the floating-point atomic min/max builtins, focusing on
// IEEE 754 corner cases: NaN, +/-infinity, +/-zero.
//
// There are three families of operations with different semantics:
// 1. fminimum/fmaximum: IEEE 754-2019 minimum/maximum
//    - Propagates NaN (any NaN input produces NaN output)
//    - Distinguishes -0 and +0 (minimum(-0, +0) = -0, maximum(-0, +0) = +0)
//
// 2. fminimumnum/fmaximumnum: IEEE 754-2019 minimumNumber/maximumNumber
//    - Propagates numbers over NaN (minimumNumber(2.0, NaN) = 2.0)
//    - Treats -0 and +0 as equivalent
//
// 3. minnum/maxnum (existing __atomic_min_fetch for floats): IEEE 754-2008
//    - Propagates numbers over NaN (minnum(2.0, NaN) = 2.0)
//    - Treats -0 and +0 as equivalent
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

// Helper to check if two floats have the same bit pattern (for +0/-0 distinction)
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
// Test fminimum_fetch and fetch_fminimum (propagates NaN, distinguishes zeros)
//===----------------------------------------------------------------------===//

void test_fminimum_float(void) {
  printf("Testing __atomic_fminimum_fetch (float)...\n");

  // Test 1: Normal values
  {
    float x = 5.0f;
    float result = __atomic_fminimum_fetch(&x, 3.0f, MO);
    assert(result == 3.0f && "fminimum(5.0, 3.0) should be 3.0");
    assert(x == 3.0f && "stored value should be 3.0");
  }

  {
    float x = 2.0f;
    float result = __atomic_fminimum_fetch(&x, 7.0f, MO);
    assert(result == 2.0f && "fminimum(2.0, 7.0) should be 2.0");
    assert(x == 2.0f && "stored value should be 2.0");
  }

  // Test 2: NaN propagation - CRITICAL: fminimum propagates NaN
  {
    float x = 1.0f;
    float result = __atomic_fminimum_fetch(&x, NAN, MO);
    assert(is_nan_f(result) && "fminimum(1.0, NaN) should be NaN");
    assert(is_nan_f(x) && "stored value should be NaN");
  }

  {
    float x = NAN;
    float result = __atomic_fminimum_fetch(&x, 1.0f, MO);
    assert(is_nan_f(result) && "fminimum(NaN, 1.0) should be NaN");
    assert(is_nan_f(x) && "stored value should be NaN");
  }

  // Test 3: Zero handling - CRITICAL: fminimum distinguishes -0 and +0
  {
    float x = 0.0f;
    float result = __atomic_fminimum_fetch(&x, neg_zero_f(), MO);
    assert(same_bits_f(result, neg_zero_f()) &&
           "fminimum(+0, -0) should be -0");
    assert(same_bits_f(x, neg_zero_f()) && "stored value should be -0");
  }

  {
    float x = neg_zero_f();
    float result = __atomic_fminimum_fetch(&x, 0.0f, MO);
    assert(same_bits_f(result, neg_zero_f()) &&
           "fminimum(-0, +0) should be -0");
    assert(same_bits_f(x, neg_zero_f()) && "stored value should be -0");
  }

  // Test 4: Infinity
  {
    float x = INFINITY;
    float result = __atomic_fminimum_fetch(&x, 1.0f, MO);
    assert(result == 1.0f && "fminimum(+inf, 1.0) should be 1.0");
  }

  {
    float x = -INFINITY;
    float result = __atomic_fminimum_fetch(&x, 1.0f, MO);
    assert(result == -INFINITY && "fminimum(-inf, 1.0) should be -inf");
  }

  // Test 5: fetch variant (returns old value)
  {
    float x = 5.0f;
    float old = __atomic_fetch_fminimum(&x, 3.0f, MO);
    assert(old == 5.0f && "fetch_fminimum should return old value");
    assert(x == 3.0f && "stored value should be 3.0");
  }

  printf("  PASSED\n");
}

void test_fmaximum_float(void) {
  printf("Testing __atomic_fmaximum_fetch (float)...\n");

  // Test 1: Normal values
  {
    float x = 5.0f;
    float result = __atomic_fmaximum_fetch(&x, 3.0f, MO);
    assert(result == 5.0f && "fmaximum(5.0, 3.0) should be 5.0");
  }

  // Test 2: NaN propagation
  {
    float x = 1.0f;
    float result = __atomic_fmaximum_fetch(&x, NAN, MO);
    assert(is_nan_f(result) && "fmaximum(1.0, NaN) should be NaN");
  }

  // Test 3: Zero handling - fmaximum(+0, -0) should be +0
  {
    float x = 0.0f;
    float result = __atomic_fmaximum_fetch(&x, neg_zero_f(), MO);
    assert(same_bits_f(result, 0.0f) && "fmaximum(+0, -0) should be +0");
  }

  {
    float x = neg_zero_f();
    float result = __atomic_fmaximum_fetch(&x, 0.0f, MO);
    assert(same_bits_f(result, 0.0f) && "fmaximum(-0, +0) should be +0");
  }

  // Test 4: Infinity
  {
    float x = INFINITY;
    float result = __atomic_fmaximum_fetch(&x, 1.0f, MO);
    assert(result == INFINITY && "fmaximum(+inf, 1.0) should be +inf");
  }

  printf("  PASSED\n");
}

//===----------------------------------------------------------------------===//
// Test fminimumnum_fetch (propagates numbers, treats zeros as equivalent)
//===----------------------------------------------------------------------===//

void test_fminimum_num_float(void) {
  printf("Testing __atomic_fminimum_num_fetch (float)...\n");

  // Test 1: Normal values
  {
    float x = 5.0f;
    float result = __atomic_fminimum_num_fetch(&x, 3.0f, MO);
    assert(result == 3.0f && "fminimumnum(5.0, 3.0) should be 3.0");
  }

  // Test 2: NaN handling - CRITICAL: fminimumnum propagates NUMBER over NaN
  {
    float x = 1.0f;
    float result = __atomic_fminimum_num_fetch(&x, NAN, MO);
    assert(result == 1.0f &&
           "fminimumnum(1.0, NaN) should be 1.0 (number over NaN)");
    assert(x == 1.0f && "stored value should be 1.0");
  }

  {
    float x = NAN;
    float result = __atomic_fminimum_num_fetch(&x, 2.0f, MO);
    assert(result == 2.0f &&
           "fminimumnum(NaN, 2.0) should be 2.0 (number over NaN)");
    assert(x == 2.0f && "stored value should be 2.0");
  }

  {
    float x = NAN;
    float result = __atomic_fminimum_num_fetch(&x, NAN, MO);
    assert(is_nan_f(result) && "fminimumnum(NaN, NaN) should be NaN");
  }

  // Test 3: Zero handling - fminimumnum treats +0 and -0 as equivalent
  // The result can be either, but should pick the minimum value
  {
    float x = 0.0f;
    float result = __atomic_fminimum_num_fetch(&x, neg_zero_f(), MO);
    // Result should be a zero (either +0 or -0 is acceptable per IEEE 754)
    assert(result == 0.0f && "fminimumnum(+0, -0) should be zero");
  }

  // Test 4: Infinity
  {
    float x = INFINITY;
    float result = __atomic_fminimum_num_fetch(&x, 1.0f, MO);
    assert(result == 1.0f && "fminimumnum(+inf, 1.0) should be 1.0");
  }

  // Test 5: fetch variant
  {
    float x = NAN;
    float old = __atomic_fetch_fminimum_num(&x, 3.0f, MO);
    assert(is_nan_f(old) && "fetch_fminimum_num should return old value (NaN)");
    assert(x == 3.0f && "stored value should be 3.0");
  }

  printf("  PASSED\n");
}

void test_fmaximum_num_float(void) {
  printf("Testing __atomic_fmaximum_num_fetch (float)...\n");

  // Test 1: Normal values
  {
    float x = 5.0f;
    float result = __atomic_fmaximum_num_fetch(&x, 3.0f, MO);
    assert(result == 5.0f && "fmaximumnum(5.0, 3.0) should be 5.0");
  }

  // Test 2: NaN handling - propagates number over NaN
  {
    float x = 1.0f;
    float result = __atomic_fmaximum_num_fetch(&x, NAN, MO);
    assert(result == 1.0f && "fmaximumnum(1.0, NaN) should be 1.0");
  }

  {
    float x = NAN;
    float result = __atomic_fmaximum_num_fetch(&x, 2.0f, MO);
    assert(result == 2.0f && "fmaximumnum(NaN, 2.0) should be 2.0");
  }

  // Test 3: Zero handling - treats +0 and -0 as equivalent
  {
    float x = 0.0f;
    float result = __atomic_fmaximum_num_fetch(&x, neg_zero_f(), MO);
    assert(result == 0.0f && "fmaximumnum(+0, -0) should be zero");
  }

  printf("  PASSED\n");
}

//===----------------------------------------------------------------------===//
// Double precision tests
//===----------------------------------------------------------------------===//

void test_fminimum_double(void) {
  printf("Testing __atomic_fminimum_fetch (double)...\n");

  // Test NaN propagation
  {
    double x = 1.0;
    double result = __atomic_fminimum_fetch(&x, NAN, MO);
    assert(is_nan_d(result) && "fminimum(1.0, NaN) should be NaN (double)");
  }

  // Test zero distinction
  {
    double x = 0.0;
    double result = __atomic_fminimum_fetch(&x, neg_zero_d(), MO);
    assert(same_bits_d(result, neg_zero_d()) &&
           "fminimum(+0, -0) should be -0 (double)");
  }

  // Test normal values
  {
    double x = 3.14;
    double result = __atomic_fminimum_fetch(&x, 2.71, MO);
    assert(result == 2.71 && "fminimum(3.14, 2.71) should be 2.71");
  }

  printf("  PASSED\n");
}

void test_fmaximum_double(void) {
  printf("Testing __atomic_fmaximum_fetch (double)...\n");

  // Test NaN propagation
  {
    double x = 1.0;
    double result = __atomic_fmaximum_fetch(&x, NAN, MO);
    assert(is_nan_d(result) && "fmaximum(1.0, NaN) should be NaN (double)");
  }

  // Test zero distinction - fmaximum(+0, -0) = +0
  {
    double x = 0.0;
    double result = __atomic_fmaximum_fetch(&x, neg_zero_d(), MO);
    assert(same_bits_d(result, 0.0) &&
           "fmaximum(+0, -0) should be +0 (double)");
  }

  printf("  PASSED\n");
}

void test_fminimum_num_double(void) {
  printf("Testing __atomic_fminimum_num_fetch (double)...\n");

  // Test number over NaN
  {
    double x = NAN;
    double result = __atomic_fminimum_num_fetch(&x, 2.5, MO);
    assert(result == 2.5 && "fminimumnum(NaN, 2.5) should be 2.5 (double)");
  }

  // Test normal values
  {
    double x = 10.5;
    double result = __atomic_fminimum_num_fetch(&x, 8.3, MO);
    assert(result == 8.3 && "fminimumnum(10.5, 8.3) should be 8.3");
  }

  printf("  PASSED\n");
}

void test_fmaximum_num_double(void) {
  printf("Testing __atomic_fmaximum_num_fetch (double)...\n");

  // Test number over NaN
  {
    double x = NAN;
    double result = __atomic_fmaximum_num_fetch(&x, 2.5, MO);
    assert(result == 2.5 && "fmaximumnum(NaN, 2.5) should be 2.5 (double)");
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

  printf("--- fminimum/fmaximum (propagate NaN, distinguish zeros) ---\n");
  test_fminimum_float();
  test_fmaximum_float();
  test_fminimum_double();
  test_fmaximum_double();

  printf("\n--- fminimumnum/fmaximumnum (prefer numbers, treat zeros equal) "
         "---\n");
  test_fminimum_num_float();
  test_fmaximum_num_float();
  test_fminimum_num_double();
  test_fmaximum_num_double();

  printf("\n");
  printf("=============================================================\n");
  printf("All tests PASSED!\n");
  printf("=============================================================\n");
  printf("\n");

  return 0;
}
