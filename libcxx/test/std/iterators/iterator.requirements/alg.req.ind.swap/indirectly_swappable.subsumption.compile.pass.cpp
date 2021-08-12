//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class I1, class I2>
// concept indirectly_swappable;

#include <iterator>

#include <concepts>

template<class I1, class I2>
  requires std::indirectly_readable<I1> && std::indirectly_readable<I2>
constexpr bool indirectly_swappable_subsumption() {
  return false;
}

template<class I1, class I2>
  requires std::indirectly_swappable<I1, I2>
constexpr bool indirectly_swappable_subsumption() {
  return true;
}

static_assert(indirectly_swappable_subsumption<int*, int*>());
