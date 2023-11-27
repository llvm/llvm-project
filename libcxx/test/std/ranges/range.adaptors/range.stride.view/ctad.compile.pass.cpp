//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::views::stride_view

#include "test.h"
#include <concepts>
#include <ranges>
#include <utility>

constexpr bool test() {
  int arr[]{1, 2, 3};

  MovedCopiedTrackedBasicView<int> bv{arr, arr + 3};
  InstrumentedBasicRange<int> br{};

  static_assert(std::same_as< decltype(std::ranges::stride_view(bv, 2)), std::ranges::stride_view<decltype(bv)> >);
  static_assert(
      std::same_as< decltype(std::ranges::stride_view(std::move(bv), 2)), std::ranges::stride_view<decltype(bv)> >);

  static_assert(std::same_as< decltype(std::ranges::drop_view(br, 0)),
                              std::ranges::drop_view<std::ranges::ref_view<InstrumentedBasicRange<int>>> >);

  static_assert(std::same_as< decltype(std::ranges::drop_view(std::move(br), 0)),
                              std::ranges::drop_view<std::ranges::owning_view<InstrumentedBasicRange<int>>> >);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
