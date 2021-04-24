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
// concept incrementable;

#include <iterator>

#include <concepts>

// clang-format off
template<std::weakly_incrementable I>
requires std::regular<I>
[[nodiscard]] constexpr bool check_subsumption() {
  return false;
}

template<std::incrementable>
[[nodiscard]] constexpr bool check_subsumption() {
  return true;
}
// clang-format on

static_assert(check_subsumption<int*>());
