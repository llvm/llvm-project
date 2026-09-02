//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// std::views::concat

#include <ranges>

#include <array>
#include <cassert>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "../range_adaptor_types.h"

static_assert(!std::is_invocable_v<decltype((std::views::concat))>);
static_assert(!std::is_invocable_v<decltype((std::views::concat)), int>);
static_assert(std::is_invocable_v<decltype((std::views::concat)), SizedRandomAccessView>);
static_assert(std::is_invocable_v<decltype((std::views::concat)), SizedRandomAccessView, NonSimpleCommon>);
static_assert(std::is_invocable_v<decltype((std::views::concat)),
                                  SizedRandomAccessView,
                                  NonSimpleCommon,
                                  NonSimpleBidiCommonView>);
static_assert(!std::is_invocable_v<decltype((std::views::concat)), SizedRandomAccessView, int>);

constexpr bool test() {
  {
    // single range
    int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::same_as<decltype(std::views::all((std::forward<SizedRandomAccessView>(buffer))))> decltype(auto) v =
        std::ranges::views::concat(SizedRandomAccessView{buffer});
    assert(std::ranges::size(v) == 8);
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(v)>, int&>);
  }

  {
    // single view as output range will be rejected
    // https://cplusplus.github.io/LWG/issue4082
    std::vector<int> v{1, 2, 3};
    static_assert(
        !std::is_invocable_v<decltype((std::views::concat)), decltype(std::views::counted(std::back_inserter(v), 3))>);
  }

  {
    // more than one ranges
    int buffer[4] = {1, 2, 3, 4};
    std::array<int, 3> a{1, 2, 3};
    std::same_as<std::ranges::concat_view<NonSimpleCommonRandomAccessSized,
                                          std::ranges::ref_view<std::array<int, 3>>>> decltype(auto) v =
        std::ranges::views::concat(NonSimpleCommonRandomAccessSized{buffer}, a);
    assert(&(*v.begin()) == &(buffer[0]));
    assert(&(*(v.begin() + 4)) == &(a[0]));
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(v)>, int&>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
