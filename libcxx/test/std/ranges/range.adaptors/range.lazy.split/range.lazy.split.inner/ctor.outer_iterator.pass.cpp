//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr explicit inner-iterator::inner-iterator(outer-iterator<Const> i);

#include <ranges>

#include "../types.h"

static_assert(!std::is_constructible_v<InnerIterNonConst, OuterIterConst>);

template <class Inner, class Outer>
constexpr void test_impl() {
  [[maybe_unused]] Inner i(Outer{});
  // Verify that the constructor is `explicit`.
  static_assert(!std::is_convertible_v<Outer, Inner>);
}

constexpr bool test() {
  test_impl<InnerIterForward, OuterIterForward>();
  test_impl<InnerIterInput, OuterIterInput>();
// Is only constructible if both the outer and the inner iterators have the same constness.
  test_impl<InnerIterConst, OuterIterConst>();
// Note: this works because of an implicit conversion (`OuterIterNonConst` is converted to `OuterIterConst`).
  test_impl<InnerIterConst, OuterIterNonConst>();
  test_impl<InnerIterNonConst, OuterIterNonConst>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
