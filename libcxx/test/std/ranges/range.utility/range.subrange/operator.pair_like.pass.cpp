
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<different-from<subrange> PairLike>
//   requires pair-like-convertible-from<PairLike, const I&, const S&>
// constexpr operator PairLike() const;

#include <cassert>
#include <concepts>
#include <ranges>

#include "test_macros.h"

static_assert(std::convertible_to<std::ranges::subrange<int*>, std::pair<int*, int*>>);
static_assert(std::convertible_to<std::ranges::subrange<int*>, std::tuple<int*, int*>>);
static_assert(!std::convertible_to<std::ranges::subrange<int*>, std::pair<long*, int*>>);
static_assert(!std::convertible_to<std::ranges::subrange<int*>, std::pair<int*, long*>>);
static_assert(!std::convertible_to<std::ranges::subrange<int*>, std::pair<long*, long*>>);
static_assert(!std::convertible_to<std::ranges::subrange<int*>, std::array<int*, 2>>);

constexpr bool test() {
  // Check to std::pair
  {
    int data[] = {1, 2, 3, 4, 5};
    const std::ranges::subrange a(data);
    {
      std::pair<int*, int*> p(a);
      assert(p.first == data + 0);
      assert(p.second == data + 5);
    }
    {
      std::pair<int*, int*> p{a};
      assert(p.first == data + 0);
      assert(p.second == data + 5);
    }
    {
      std::pair<int*, int*> p = a;
      assert(p.first == data + 0);
      assert(p.second == data + 5);
    }
  }

  // Check to std::tuple
  {
    int data[] = {1, 2, 3, 4, 5};
    const std::ranges::subrange a(data);
    {
      std::tuple<int*, int*> p(a);
      assert(std::get<0>(p) == data + 0);
      assert(std::get<1>(p) == data + 5);
    }
    {
      std::tuple<int*, int*> p{a};
      assert(std::get<0>(p) == data + 0);
      assert(std::get<1>(p) == data + 5);
    }
    {
      std::tuple<int*, int*> p = a;
      assert(std::get<0>(p) == data + 0);
      assert(std::get<1>(p) == data + 5);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
