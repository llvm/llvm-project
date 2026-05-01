//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// constexpr reference operator[](difference_type n) const
//   requires cartesian-product-is-random-access<...>;

#include <array>
#include <cassert>
#include <ranges>

#include "../../range_adaptor_types.h"

constexpr bool test() {
  std::array a{1, 2, 3, 4, 5, 6, 7, 8};

  { // single random_access range
    std::ranges::cartesian_product_view v(SizedRandomAccessView{a});
    auto it = v.begin();
    assert(it[0] == *it);
    assert(it[3] == *(it + 3));
    assert(it[7] == *(it + 7));
    static_assert(std::is_same_v<decltype(it[2]), std::tuple<int&>>);
  }

  { // 2 random_access ranges -- operator[] returns reference tuple
    std::ranges::cartesian_product_view v(SizedRandomAccessView{a}, std::views::iota(0, 5));
    auto it = v.begin();
    assert(it[0] == *it);
    assert(it[5] == *(it + 5));
    static_assert(std::is_same_v<decltype(it[2]), std::tuple<int&, int>>);
  }

  { // contiguous ranges
    std::ranges::cartesian_product_view v(ContiguousCommonView{a}, ContiguousCommonView{a});
    auto it = v.begin();
    assert(it[0] == *it);
    assert(it[10] == *(it + 10));
    static_assert(std::is_same_v<decltype(it[2]), std::tuple<int&, int&>>);
  }

  { // not random_access -- operator[] should not be available
    std::ranges::cartesian_product_view v(BidiCommonView{a});
    auto it                  = v.begin();
    const auto can_subscript = [](auto&& i) { return requires { i[0]; }; };
    static_assert(!can_subscript(it));
  }

  { // forward+sized -- also not random_access for cartesian
    std::ranges::cartesian_product_view v(ForwardSizedView{a});
    auto it                  = v.begin();
    const auto can_subscript = [](auto&& i) { return requires { i[0]; }; };
    static_assert(!can_subscript(it));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
