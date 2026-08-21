//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// std::views::cartesian_product
//   * Zero-argument form returns views::single(tuple()).
//   * N-argument form returns cartesian_product_view<all_t<R>...>.
//   * Both forms are expression-equivalent to those expressions, so they agree on potentially-throwing-ness.

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

static_assert(std::same_as<decltype(std::views::cartesian_product), decltype(std::ranges::views::cartesian_product)>);

// [range.cartesian.overview]/2 makes views::cartesian_product(Es...) expression-equivalent to the expression it
// returns, and [defns.expression-equivalent] requires expression-equivalent expressions to be either all
// potentially-throwing or all not. Neither single_view's nor cartesian_product_view's constructor is noexcept, so
// both forms are potentially-throwing today; the comparisons below keep holding if that ever changes.
static_assert(!noexcept(std::views::cartesian_product()));
static_assert(noexcept(std::views::cartesian_product()) == noexcept(std::views::single(std::tuple())));

static_assert(!noexcept(std::views::cartesian_product(std::declval<SimpleCommon&>())));
static_assert(
    !noexcept(std::views::cartesian_product(std::declval<SimpleCommon&>(), std::declval<ForwardSizedView&>())));
static_assert(noexcept(std::views::cartesian_product(std::declval<SimpleCommon&>())) ==
              noexcept(std::ranges::cartesian_product_view<SimpleCommon>(std::declval<SimpleCommon&>())));

constexpr bool test() {
  {
    // zero arguments: produces views::single(tuple()) -- a single-element view of an empty tuple
    std::same_as<std::ranges::single_view<std::tuple<>>> decltype(auto) v = std::views::cartesian_product();
    static_assert(std::ranges::sized_range<decltype(v)>);
    assert(v.size() == 1);

    auto it = v.begin();
    assert(it != v.end());
    static_assert(std::same_as<decltype(*it), std::tuple<>&>);
    ++it;
    assert(it == v.end());
  }

  {
    // a single view
    int buffer[3] = {1, 2, 3};
    std::same_as<std::ranges::cartesian_product_view<SizedRandomAccessView>> decltype(auto) v =
        std::views::cartesian_product(SizedRandomAccessView{buffer});
    assert(v.size() == 3);
    assert(*v.begin() == std::tuple<int&>(buffer[0]));
  }

  {
    // more than one range, each forwarded through views::all
    int buffer[2] = {1, 2};
    std::same_as<std::ranges::cartesian_product_view<std::ranges::ref_view<int[2]>,
                                                     std::ranges::iota_view<int, int>>> decltype(auto) v =
        std::views::cartesian_product(buffer, std::views::iota(0, 4));
    assert(v.size() == 8);
    assert(&(std::get<0>(*v.begin())) == &(buffer[0]));
  }

  {
    // a moved-in range is wrapped in an owning_view
    int buffer[3] = {10, 20, 30};
    std::same_as<std::ranges::cartesian_product_view<std::ranges::owning_view<std::array<int, 2>>,
                                                     std::ranges::ref_view<int[3]>>> decltype(auto) v =
        std::views::cartesian_product(std::array{1, 2}, buffer);
    assert(v.size() == 6);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
