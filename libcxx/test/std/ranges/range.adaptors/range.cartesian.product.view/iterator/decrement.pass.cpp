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
#include <concepts>
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
    std::ranges::cartesian_product_view v(BidiCommonView{a}, BidiNonCommonView{b});
    using Iter = std::ranges::iterator_t<decltype(v)>;
    static_assert(!CanDecrement<Iter>);
  }

  { // non-simple bases -- -- must instantiate __iterator<false>; __prev calls
    // __cartesian_common_arg_end on a *non-const* base
    std::ranges::cartesian_product_view v(NonSimpleCommon{a}, NonSimpleCommon{b});
    // Pin the specialisation: a later switch to a simple view would silently drop this coverage.
    static_assert(!std::same_as<decltype(v.begin()), std::ranges::iterator_t<const decltype(v)>>);

    auto it    = v.end();
    using Iter = decltype(it);
    static_assert(std::is_same_v<decltype(--it), Iter&>);
    static_assert(std::is_same_v<decltype(it--), Iter>);

    --it;
    assert(*it == std::tuple(3, 20));
    --it;
    assert(*it == std::tuple(3, 10));
    --it; // borrow from a
    assert(*it == std::tuple(2, 20));

    auto copy = it--;
    assert(*copy == std::tuple(2, 20));
    assert(*it == std::tuple(2, 10));
    --it; // borrow from a
    assert(*it == std::tuple(1, 20));
    --it;
    assert(*it == std::tuple(1, 10));
    assert(it == v.begin());
  }

  { // non-simple bases, 3 ranges -- reaches the intermediate __prev<1> recursion
    std::array c{100, 200};
    std::ranges::cartesian_product_view v(NonSimpleCommon{a}, NonSimpleCommon{b}, NonSimpleCommon{c});
    static_assert(!std::same_as<decltype(v.begin()), std::ranges::iterator_t<const decltype(v)>>);

    auto it = v.end();
    --it;
    assert(*it == std::tuple(3, 20, 200));
    --it;
    assert(*it == std::tuple(3, 20, 100));
    --it; // borrow from b
    assert(*it == std::tuple(3, 10, 200));
    --it;
    assert(*it == std::tuple(3, 10, 100));
    --it; // borrow through b from a
    assert(*it == std::tuple(2, 20, 200));
  }

  { // LWG3820: "cartesian_product_view::iterator::prev is not quite right".
    // `prev` must not apply cartesian-common-arg-end to the first range, which is required to
    // model neither common_range nor sized_range. This is the example from the issue.
    auto v  = std::views::cartesian_product(std::views::iota(0));
    auto it = v.begin() + 3;
    assert(*it == std::tuple(3));

    --it;
    assert(*it == std::tuple(2));

    assert(*it-- == std::tuple(2));
    assert(*it == std::tuple(1));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
