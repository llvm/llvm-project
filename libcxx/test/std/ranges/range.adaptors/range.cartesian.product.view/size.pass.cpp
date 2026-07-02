//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// constexpr auto size()       requires cartesian-product-is-sized<      First,       Vs...>;
// constexpr auto size() const requires cartesian-product-is-sized<const First, const Vs...>;

#include <array>
#include <cassert>
#include <concepts>
#include <initializer_list>
#include <ranges>

#include "../range_adaptor_types.h"

template <class T>
concept HasSize = requires(T&& t) { t.size(); };

constexpr bool test() {
  { // example from cppreference
    constexpr static auto w = {1};
    constexpr static auto x = {2, 3};
    constexpr static auto y = {4, 5, 6};
    constexpr static auto z = {7, 8, 9, 10, 11, 12, 13};

    constexpr auto v = std::ranges::cartesian_product_view(
        std::views::all(w), std::views::all(x), std::views::all(y), std::views::all(z));

    assert(v.size() == 42);
    assert(v.size() == w.size() * x.size() * y.size() * z.size());
  }

  { // empty range yields size 0
    std::ranges::empty_view<int> e;
    auto v = std::ranges::cartesian_product_view(e);
    assert(v.size() == 0);
  }

  { // 1-3 ranges
    constexpr std::size_t N0 = 3, N1 = 7, N2 = 42;
    std::array<int, N0> a0{};
    std::array<int, N1> a1{};
    std::array<int, N2> a2{};
    assert(std::ranges::cartesian_product_view(a0).size() == N0);
    assert(std::ranges::cartesian_product_view(a0, a1).size() == N0 * N1);
    assert(std::ranges::cartesian_product_view(a0, a1, a2).size() == N0 * N1 * N2);
  }

  { // size() return type -- common_type of underlying range_size_t
    std::array<int, 3> a;
    auto v = std::ranges::cartesian_product_view(a);
    static_assert(std::unsigned_integral<decltype(v.size())>);
  }

  return true;
}

// Negative case: an unsized range disables size() on the cartesian product.
// (NonSizedRandomAccessView from range_adaptor_types.h is random-access but not sized.)
static_assert(!std::ranges::sized_range<NonSizedRandomAccessView>);
static_assert(!HasSize<std::ranges::cartesian_product_view<NonSizedRandomAccessView>>);
static_assert(!HasSize<std::ranges::cartesian_product_view<SimpleCommon, NonSizedRandomAccessView>>);

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
