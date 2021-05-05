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

// template<class T>
// concept input_iterator;

#include <iterator>

#include <concepts>

// clang-format off
template<std::input_or_output_iterator I>
requires std::indirectly_readable<I>
[[nodiscard]] constexpr bool check_subsumption() {
  return false;
}

template<std::input_iterator>
[[nodiscard]] constexpr bool check_subsumption() {
  return true;
}
// clang-format on

static_assert(check_subsumption<int*>());
