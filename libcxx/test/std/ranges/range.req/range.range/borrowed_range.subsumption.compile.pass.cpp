//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class T>
// concept borrowed_range;

#include <ranges>

template <std::ranges::range R>
consteval bool check_subsumption() {
  return false;
}

template <std::ranges::borrowed_range R>
consteval bool check_subsumption() {
  return true;
}

static_assert(check_subsumption<int (&)[8]>());
