//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: apple-clang-12

// In MSVC mode, there's a slightly different number of errors printed for
// each of these, so it doesn't add up to the exact expected count of 18.
// XFAIL: msvc

// <compare>

// Ensure we reject all cases where an argument other than a literal 0 is used
// for a comparison against a comparison category type.

#include <compare>

#define TEST_FAIL(v, op)                                                       \
  void(v op 0L);                                                               \
  void(0L op v);                                                               \
  void(v op nullptr);                                                          \
  void(nullptr op v);                                                          \
  void(v op(1 - 1));                                                           \
  void((1 - 1) op v)

#define TEST_PASS(v, op)                                                       \
  void(v op 0);                                                                \
  void(0 op v)

template <typename T>
void test_category(T v) {
  TEST_FAIL(v, ==);  // expected-error 18 {{}}
  TEST_FAIL(v, !=);  // expected-error 18 {{}}
  TEST_FAIL(v, <);   // expected-error 18 {{}}
  TEST_FAIL(v, <=);  // expected-error 18 {{}}
  TEST_FAIL(v, >);   // expected-error 18 {{}}
  TEST_FAIL(v, >=);  // expected-error 18 {{}}
  TEST_FAIL(v, <=>); // expected-error 18 {{}}

  TEST_PASS(v, ==);
  TEST_PASS(v, !=);
  TEST_PASS(v, <);
  TEST_PASS(v, >);
  TEST_PASS(v, <=);
  TEST_PASS(v, >=);
  TEST_PASS(v, <=>);
}

int main(int, char**) {
  test_category(std::strong_ordering::equivalent);
  test_category(std::weak_ordering::equivalent);
  test_category(std::partial_ordering::equivalent);
  return 0;
}
