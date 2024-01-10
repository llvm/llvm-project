//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <numeric>

// template<class R, class T>
//   constexpr R saturate_cast(T x) noexcept;                    // freestanding

#include <cstdint>
#include <numeric>

#include "test_macros.h"

template <typename ResultIntegerT, typename IntegerT>
constexpr void test_constraint() {
  // expected-error-re@*:* 0-2 {{constant expression evaluates to {{.*}} which cannot be narrowed to type {{.*}}}}
  // expected-error@*:* 0-6 {{no matching function for call to 'saturate_cast'}}
  // expected-error@*:* 0-4 {{expected unqualified-id}}
  [[maybe_unused]] auto sum = std::saturate_cast<ResultIntegerT>(IntegerT{4});
}

constexpr bool test() {
  test_constraint<bool, int>();
  test_constraint<char, int>();
  test_constraint<int, bool>();
  test_constraint<int, char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_constraint<wchar_t, int>();
  test_constraint<int, wchar_t>();
#endif
  test_constraint<std::char16_t, int>();
  test_constraint<std::char32_t, int>();
  test_constraint<int, std::char16_t>();
  test_constraint<int, std::char32_t>();

  return true;
}
