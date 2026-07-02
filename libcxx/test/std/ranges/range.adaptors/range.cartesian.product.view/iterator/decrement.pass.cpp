//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// constexpr iterator& operator--() requires cartesian-product-is-bidirectional<...>;
// constexpr iterator operator--(int) requires cartesian-product-is-bidirectional<...>;

#include <array>
#include <cassert>
#include <ranges>
#include <tuple>

#include "../../range_adaptor_types.h"

template <class Iter>
concept CanDecrement = requires(Iter it) { --it; } || requires(Iter it) { it--; };

constexpr bool test() {
  std::array a{1, 2, 3};
  std::array b{10, 20};

  { // random-access -- decrementing v.end() yields the last element
    std::ranges::cartesian_product_view v(a, b);
    auto it    = v.end();
    using Iter = decltype(it);

    static_assert(std::is_same_v<decltype(--it), Iter&>);
    auto& it_ref = --it;
    assert(&it_ref == &it);

    // Last element is (3, 20)
    assert(*it == std::tuple(3, 20));

    static_assert(std::is_same_v<decltype(it--), Iter>);
    auto copy = it--;
    assert(*copy == std::tuple(3, 20));
    assert(*it == std::tuple(3, 10));
  }

  { // 3-range -- wraparound through two levels going backwards
    std::array c{100, 200};
    std::ranges::cartesian_product_view v(a, b, c);
    auto it = v.end();
    --it;
    assert(*it == std::tuple(3, 20, 200));
    --it;
    assert(*it == std::tuple(3, 20, 100));
    --it;
    assert(*it == std::tuple(3, 10, 200));
    --it;
    assert(*it == std::tuple(3, 10, 100));
    --it;
    assert(*it == std::tuple(2, 20, 200));
  }

  { // bidi first range -- the result is bidi if every range is bidi-and-common-arg
    std::ranges::cartesian_product_view v(BidiCommonView{a});
    auto it    = v.end();
    using Iter = decltype(it);
    static_assert(CanDecrement<Iter>);

    --it;
    assert(*it == std::tuple(3));
    --it;
    assert(*it == std::tuple(2));
  }

  { // forward-only first range -- not bidirectional
    std::ranges::cartesian_product_view v(ForwardSizedView{a});
    using Iter = std::ranges::iterator_t<decltype(v)>;
    static_assert(!CanDecrement<Iter>);
  }

  { // bidi second range that is not common-arg -- not bidirectional
    using NonCommonBidiView = BidiNonCommonView;
    static_assert(!std::ranges::__cartesian_product_common_arg<NonCommonBidiView>);
    std::ranges::cartesian_product_view v(BidiCommonView{a}, NonCommonBidiView{b});
    using Iter = std::ranges::iterator_t<decltype(v)>;
    static_assert(!CanDecrement<Iter>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
