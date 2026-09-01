//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// constexpr iterator<false> end()
//   requires (!simple-view<First> || ... || !simple-view<Vs>)
//         && cartesian-product-is-common<First, Vs...>;
// constexpr iterator<true>  end() const
//   requires cartesian-product-is-common<const First, const Vs...>;
// constexpr default_sentinel_t end() const noexcept; // fallback

#include <array>
#include <cassert>
#include <ranges>
#include <utility>

#include "../range_adaptor_types.h"

constexpr bool test() {
  { // single non-empty range: end() == begin() + size()
    constexpr std::size_t N = 7;
    std::array<int, N> a{};
    std::ranges::cartesian_product_view c{a};
    assert(c.end() == c.begin() + N);
  }

  { // 2-range non-empty: end() == begin() + product-of-sizes
    std::array<int, 7> a0{};
    std::array<int, 42> a1{};
    std::ranges::cartesian_product_view c{a0, a1};
    assert(c.end() == c.begin() + (a0.size() * a1.size()));
  }

  { // 3-range non-empty
    std::array<int, 5> a0{};
    std::array<int, 42> a1{};
    std::array<int, 7> a2{};
    std::ranges::cartesian_product_view c{a0, a1, a2};
    assert(c.end() == c.begin() + (a0.size() * a1.size() * a2.size()));
  }

  { // when ANY range is empty, the entire product is empty: begin() == end()
    std::ranges::empty_view<int> e;
    constexpr std::size_t N = 7;
    std::array<int, N> a{};

    {
      std::ranges::cartesian_product_view c{e};
      assert(c.end() == c.begin());
    }
    {
      std::ranges::cartesian_product_view c{e, e};
      assert(c.end() == c.begin());
    }
    {
      std::ranges::cartesian_product_view c{e, e, e};
      assert(c.end() == c.begin());
    }

    // empty in any position of a 2-range product
    {
      std::ranges::cartesian_product_view c{e, a};
      assert(c.end() == c.begin());
    }
    {
      std::ranges::cartesian_product_view c{a, e};
      assert(c.end() == c.begin());
    }

    // empty in any position of a 3-range product
    {
      std::ranges::cartesian_product_view c{e, a, a};
      assert(c.end() == c.begin());
    }
    {
      std::ranges::cartesian_product_view c{a, e, a};
      assert(c.end() == c.begin());
    }
    {
      std::ranges::cartesian_product_view c{a, a, e};
      assert(c.end() == c.begin());
    }

    // multiple empties
    {
      std::ranges::cartesian_product_view c{e, e, a};
      assert(c.end() == c.begin());
    }
    {
      std::ranges::cartesian_product_view c{a, e, e};
      assert(c.end() == c.begin());
    }
  }

  { // simple-view: const and non-const end() return the same iterator type
    std::array<int, 3> a{};
    std::ranges::cartesian_product_view v{SimpleCommon{a.data(), a.size()}};
    static_assert(std::same_as<decltype(v.end()), decltype(std::as_const(v).end())>);
  }

  { // non-simple-view: const and non-const end() differ
    std::array<int, 3> a{};
    std::ranges::cartesian_product_view v{NonSimpleCommon{a.data(), a.size()}};
    static_assert(!std::same_as<decltype(v.end()), decltype(std::as_const(v).end())>);
    assert(v.end() == std::as_const(v).end());
  }

  { // non-common first range falls back to default_sentinel_t
    int data[3] = {1, 2, 3};
    std::ranges::cartesian_product_view v{InputNonCommonView{data}};
    using View = decltype(v);
    static_assert(!std::ranges::common_range<View>);
    static_assert(std::same_as<decltype(v.end()), std::default_sentinel_t>);
    static_assert(std::same_as<decltype(std::as_const(v).end()), std::default_sentinel_t>);
    static_assert(noexcept(std::as_const(v).end()));
  }

  { // sized + random_access first range is also a common-arg even when not common_range
    int data[3] = {1, 2, 3};
    std::ranges::cartesian_product_view v{ContiguousNonCommonSized{data}};
    using View = decltype(v);
    static_assert(std::ranges::common_range<View>);
    // end() returns an iterator (not default_sentinel_t).
    static_assert(!std::same_as<decltype(v.end()), std::default_sentinel_t>);
    assert(v.end() == v.begin() + 3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
