//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <compare>

// Ensure we reject all cases where an argument other than a literal 0 is used
// for a comparison against a comparison category type.

// Also ensure that we don't warn about providing a null pointer constant when
// comparing an ordering type against literal 0, since one of the common
// implementation strategies is to use a pointer as the "unspecified type".
// ADDITIONAL_COMPILE_FLAGS: -Wzero-as-null-pointer-constant

#include <compare>

#include "test_macros.h"

#define TEST_FAIL(v, op)                                                                                               \
  do {                                                                                                                 \
    /* invalid types */                                                                                                \
    void(v op 0L);                                                                                                     \
    void(0L op v);                                                                                                     \
    void(v op 0.0);                                                                                                    \
    void(0.0 op v);                                                                                                    \
    void(v op nullptr);                                                                                                \
    void(nullptr op v);                                                                                                \
    /* invalid value */                                                                                                \
    void(v op 1);                                                                                                      \
    void(1 op v);                                                                                                      \
    /* value not known at compile-time */                                                                              \
    int i = 0;                                                                                                         \
    void(v op i);                                                                                                      \
    void(i op v);                                                                                                      \
  } while (false)

#define TEST_PASS(v, op)                                                                                               \
  do {                                                                                                                 \
    void(v op 0);                                                                                                      \
    void(0 op v);                                                                                                      \
    LIBCPP_ONLY(void(v op(1 - 1)));                                                                                    \
    LIBCPP_ONLY(void((1 - 1) op v));                                                                                   \
  } while (false)

template <typename T>
void test_category(T v) {
  TEST_FAIL(v, ==);  // expected-error 30 {{invalid operands to binary expression}}
  TEST_FAIL(v, !=);  // expected-error 30 {{invalid operands to binary expression}}
  TEST_FAIL(v, <);   // expected-error 30 {{invalid operands to binary expression}}
  TEST_FAIL(v, <=);  // expected-error 30 {{invalid operands to binary expression}}
  TEST_FAIL(v, >);   // expected-error 30 {{invalid operands to binary expression}}
  TEST_FAIL(v, >=);  // expected-error 30 {{invalid operands to binary expression}}
  TEST_FAIL(v, <=>); // expected-error 30 {{invalid operands to binary expression}}

  TEST_PASS(v, ==);
  TEST_PASS(v, !=);
  TEST_PASS(v, <);
  TEST_PASS(v, >);
  TEST_PASS(v, <=);
  TEST_PASS(v, >=);
  TEST_PASS(v, <=>);
}

void f() {
  test_category(std::strong_ordering::equivalent);
  test_category(std::weak_ordering::equivalent);
  test_category(std::partial_ordering::equivalent);
}
