//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template<class T>
// concept approximately_sized_range;

#include <ranges>

// approximately_sized_range subsumes range

template <std::ranges::range R>
consteval bool check_subsumption_1() {
  return false;
}

template <std::ranges::approximately_sized_range R>
consteval bool check_subsumption_1() {
  return true;
}

static_assert(check_subsumption_1<int[5]>());

// sized_range subsumes approximately_sized_range

template <std::ranges::approximately_sized_range R>
consteval bool check_subsumption_2() {
  return false;
}

template <std::ranges::sized_range R>
consteval bool check_subsumption_2() {
  return true;
}

static_assert(check_subsumption_2<int[5]>());
