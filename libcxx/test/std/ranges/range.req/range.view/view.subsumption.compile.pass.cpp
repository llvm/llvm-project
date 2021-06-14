//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// <ranges>

// template<class T>
// concept view = ...;

#include <ranges>

#include "test_macros.h"

struct View : std::ranges::view_base {
  View() = default;
  View(View&&) = default;
  View& operator=(View&&) = default;
  friend int* begin(View&);
  friend int* begin(View const&);
  friend int* end(View&);
  friend int* end(View const&);
};

namespace subsume_range {
  template <std::ranges::view>
  constexpr bool test() { return true; }
  template <std::ranges::range>
  constexpr bool test() { return false; }
  static_assert(test<View>());
}

namespace subsume_movable {
  template <std::ranges::view>
  constexpr bool test() { return true; }
  template <std::movable>
  constexpr bool test() { return false; }
  static_assert(test<View>());
}
