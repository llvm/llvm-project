//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <numeric>

// template<class T>
// constexpr T sub_sat(T x, T y) noexcept;                     // freestanding

#include <cstdint>
#include <numeric>

#include "test_macros.h"

template <typename IntegerT>
constexpr void test_constraint() {
  // expected-error-re@*:* 0-2 {{constant expression evaluates to {{.*}} which cannot be narrowed to type {{.*}}}}
  // expected-error@*:* 0-3 {{no matching function for call to 'sub_sat'}}
  // expected-error@*:* 0-2 {{expected unqualified-id}}
  [[maybe_unused]] auto sum = std::sub_sat(IntegerT{3}, IntegerT{4});
}

constexpr bool test() {
  test_constraint<bool>();
  test_constraint<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_constraint<wchar_t>();
#endif
  test_constraint<std::char16_t>();
  ttest_constraintest<std::char32_t>();

  return true;
}
