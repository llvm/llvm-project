//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// friend constexpr difference_type operator-(const iterator& i, default_sentinel_t)
//   requires cartesian-is-sized-sentinel<Const, sentinel_t, First, Vs...>;
// friend constexpr difference_type operator-(default_sentinel_t, const iterator& i)
//   requires cartesian-is-sized-sentinel<Const, sentinel_t, First, Vs...>;

#include <array>
#include <cassert>
#include <concepts>
#include <functional>
#include <ranges>

#include "../../range_adaptor_types.h"

template <class T, class U>
concept HasMinus = std::invocable<std::minus<>, const T&, const U&>;

constexpr bool test() {
  std::array a{1, 2, 3};
  std::array b{10, 20};

  { // single non-common but sized first range
    std::ranges::cartesian_product_view v(ForwardSizedNonCommon{a});
    static_assert(!std::ranges::common_range<decltype(v)>);
    auto it = v.begin();

    assert(std::default_sentinel - it == 3);
    assert(it - std::default_sentinel == -3);

    ++it;
    assert(std::default_sentinel - it == 2);
    assert(it - std::default_sentinel == -2);

    ++it;
    ++it;
    assert(std::default_sentinel - it == 0);
    assert(it - std::default_sentinel == 0);
  }

  { // two ranges -- distance is the Cartesian distance (product of remaining elements)
    std::ranges::cartesian_product_view v(ForwardSizedNonCommon{a}, ForwardSizedView{b});
    static_assert(!std::ranges::common_range<decltype(v)>);
    auto it = v.begin();

    assert(std::default_sentinel - it == 6);
    assert(it - std::default_sentinel == -6);

    ++it; // (a[0], b[1])
    assert(std::default_sentinel - it == 5);
    assert(it - std::default_sentinel == -5);

    ++it; // wrap inner, advance outer -> (a[1], b[0])
    assert(std::default_sentinel - it == 4);
    assert(it - std::default_sentinel == -4);
  }

  { // SFINAE: underlying first range has no sized sentinel -> no operator- with default_sentinel
    std::ranges::cartesian_product_view v(InputCommonView{a}, ForwardSizedView{b});
    using Iter = std::ranges::iterator_t<decltype(v)>;
    static_assert(!HasMinus<Iter, std::default_sentinel_t>);
    static_assert(!HasMinus<std::default_sentinel_t, Iter>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
