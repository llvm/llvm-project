//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// iterator() = default;

#include <ranges>
#include <tuple>

#include "../helpers.h"
#include "../../range_adaptor_types.h"

struct IterDefaultCtrView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

template <std::size_t N, class Fn>
constexpr void test() {
  using View = std::ranges::adjacent_transform_view<IterDefaultCtrView, MakeTuple, N>;
  using Iter = std::ranges::iterator_t<View>;
  {
    Iter iter1;
    Iter iter2 = {};
    assert(iter1 == iter2);
  }
}

template <std::size_t N>
constexpr void test() {
  test<N, MakeTuple>();
  test<N, Tie>();
  test<N, GetFirst>();
  test<N, Multiply>();
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
