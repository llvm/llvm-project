//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// std::views::cartesian_product

#include <array>
#include <cassert>
#include <concepts>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../range_adaptor_types.h"

// Output-only views are not valid ranges for the product, so the CPO SFINAEs out on them.
struct OutputOnly : std::ranges::view_base {
  cpp17_output_iterator<int*> begin() const;
  sentinel_wrapper<cpp17_output_iterator<int*>> end() const;
};

static_assert(std::is_invocable_v<decltype((std::views::cartesian_product))>);
static_assert(!std::is_invocable_v<decltype((std::views::cartesian_product)), int>);
static_assert(std::is_invocable_v<decltype((std::views::cartesian_product)), SimpleCommon>);
static_assert(std::is_invocable_v<decltype((std::views::cartesian_product)), SimpleCommon, ForwardSizedView>);
static_assert(!std::is_invocable_v<decltype((std::views::cartesian_product)), SimpleCommon, int>);
static_assert(!std::is_invocable_v<decltype((std::views::cartesian_product)), OutputOnly>);

// views::cartesian_product is a customization point object, not a range adaptor object, so it is not pipeable.
static_assert(!CanBePiped<SimpleCommon&, decltype((std::views::cartesian_product))>);
static_assert(!CanBePiped<int (&)[10], decltype((std::views::cartesian_product))>);

static_assert(std::same_as<decltype(std::views::cartesian_product), decltype(std::ranges::views::cartesian_product)>);

// views::cartesian_product(args) is expression-equivalent to the expression it returns.
static_assert(!noexcept(std::views::cartesian_product()));
static_assert(noexcept(std::views::cartesian_product()) == noexcept(std::views::single(std::tuple())));

static_assert(!noexcept(std::views::cartesian_product(std::declval<SimpleCommon&>())));
static_assert(
    !noexcept(std::views::cartesian_product(std::declval<SimpleCommon&>(), std::declval<ForwardSizedView&>())));
static_assert(noexcept(std::views::cartesian_product(std::declval<SimpleCommon&>())) ==
              noexcept(std::ranges::cartesian_product_view<SimpleCommon>(std::declval<SimpleCommon&>())));

constexpr bool test() {
  // `views::cartesian_product()` returns `views::single(tuple())`.
  {
    std::same_as<std::ranges::single_view<std::tuple<>>> decltype(auto) v = std::views::cartesian_product();
    static_assert(std::ranges::sized_range<decltype(v)>);
    assert(v.size() == 1);

    auto it = v.begin();
    assert(it != v.end());
    static_assert(std::same_as<decltype(*it), std::tuple<>&>);
    ++it;
    assert(it == v.end());
  }

  // `views::cartesian_product(view)` returns a `cartesian_product_view` of that view.
  {
    int buffer[3] = {1, 2, 3};
    std::same_as<std::ranges::cartesian_product_view<SizedRandomAccessView>> decltype(auto) v =
        std::views::cartesian_product(SizedRandomAccessView{buffer});
    assert(v.size() == 3);
    assert(*v.begin() == std::tuple<int&>(buffer[0]));
  }

  // `views::cartesian_product(rs...)` returns a `cartesian_product_view` of `views::all_t<decltype((rs))>...`.
  {
    int buffer[2] = {1, 2};
    std::same_as<std::ranges::cartesian_product_view<std::ranges::ref_view<int[2]>,
                                                     std::ranges::iota_view<int, int>>> decltype(auto) v =
        std::views::cartesian_product(buffer, std::views::iota(0, 4));
    assert(v.size() == 8);
    assert(&(std::get<0>(*v.begin())) == &(buffer[0]));
  }

  // `views::cartesian_product(rvalue, lvalue)` returns a `cartesian_product_view` over an `owning_view` and a
  // `ref_view`.
  {
    int buffer[3] = {10, 20, 30};
    std::same_as<std::ranges::cartesian_product_view<std::ranges::owning_view<std::array<int, 2>>,
                                                     std::ranges::ref_view<int[3]>>> decltype(auto) v =
        std::views::cartesian_product(std::array{1, 2}, buffer);
    assert(v.size() == 6);
  }

  // `views::cartesian_product(cartesian_product_view)` returns a nested `cartesian_product_view`.
  {
    int buffer[2] = {1, 2};
    std::same_as<std::ranges::cartesian_product_view<SizedRandomAccessView, SizedRandomAccessView>> decltype(auto) v =
        std::views::cartesian_product(SizedRandomAccessView{buffer}, SizedRandomAccessView{buffer});

    std::same_as<std::ranges::cartesian_product_view<
        std::ranges::cartesian_product_view<SizedRandomAccessView, SizedRandomAccessView>>> decltype(auto) v2 =
        std::views::cartesian_product(v);

    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(v2)>, std::tuple<std::tuple<int&, int&>>>);
    assert(v2.size() == 4);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
