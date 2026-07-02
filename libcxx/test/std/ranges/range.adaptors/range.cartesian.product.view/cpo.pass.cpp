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

#include <array>
#include <cassert>
#include <concepts>
#include <ranges>
#include <tuple>

#include "../range_adaptor_types.h"

constexpr bool test() {
  { // zero-argument: produces views::single(tuple()) -- a single-element view of an empty tuple.
    auto v   = std::views::cartesian_product();
    using V  = decltype(v);
    using ST = std::ranges::single_view<std::tuple<>>;
    static_assert(std::same_as<V, ST>);
    static_assert(std::ranges::sized_range<V>);
    assert(v.size() == 1);
    auto it = v.begin();
    assert(it != v.end());
    static_assert(std::same_as<decltype(*it), std::tuple<>&>);
    ++it;
    assert(it == v.end());
  }

  { // single-argument: returns a cartesian_product_view of all_t<R>
    int buffer[3] = {1, 2, 3};
    auto v        = std::views::cartesian_product(SizedRandomAccessView{buffer});
    static_assert(std::same_as<decltype(v), std::ranges::cartesian_product_view<SizedRandomAccessView>>);
    assert(v.size() == 3);
    assert(*v.begin() == std::tuple<int&>(buffer[0]));
  }

  { // multi-argument: forwards each range through views::all
    int b1[2] = {1, 2};
    int b2[3] = {10, 20, 30};
    auto v    = std::views::cartesian_product(b1, std::views::iota(0, 4));
    static_assert(
        std::same_as<
            decltype(v),
            std::ranges::cartesian_product_view<std::ranges::ref_view<int[2]>, std::ranges::iota_view<int, int>>>);
    assert(v.size() == 8);

    // CPO with a moved-in range produces an owning_view wrapper.
    auto v2 = std::views::cartesian_product(std::array{1, 2}, b2);
    static_assert(std::same_as<decltype(v2),
                               std::ranges::cartesian_product_view<std::ranges::owning_view<std::array<int, 2>>,
                                                                   std::ranges::ref_view<int[3]>>>);
    assert(v2.size() == 6);
  }

  return true;
}

// The CPO is invocable with zero arguments and with any combination of valid ranges.
static_assert(std::is_invocable_v<decltype(std::views::cartesian_product)>);
static_assert(std::is_invocable_v<decltype(std::views::cartesian_product), SimpleCommon>);
static_assert(std::is_invocable_v<decltype(std::views::cartesian_product), SimpleCommon, ForwardSizedView>);

// Output-only views are not valid first ranges, so the CPO SFINAEs out.
struct OutputOnly : std::ranges::view_base {
  cpp17_output_iterator<int*> begin() const;
  sentinel_wrapper<cpp17_output_iterator<int*>> end() const;
};
static_assert(!std::is_invocable_v<decltype(std::views::cartesian_product), OutputOnly>);

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
