//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// XFAIL: msvc && clang

// template<class T>
// concept input_iterator;

#include <ranges>

#include <iterator>

struct range {
  int* begin();
  int* end();
};

// clang-format off
template<std::ranges::range R>
requires std::input_iterator<std::ranges::iterator_t<R> >
[[nodiscard]] constexpr bool check_input_range_subsumption() {
  return false;
}

template<std::ranges::input_range>
requires true
[[nodiscard]] constexpr bool check_input_range_subsumption() {
  return true;
}
// clang-format on

static_assert(check_input_range_subsumption<range>());
