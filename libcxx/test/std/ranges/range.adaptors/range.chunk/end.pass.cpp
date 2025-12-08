//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::views::chunk

#include <ranges>

#include <algorithm>
#include <cassert>
#include <concepts>
#include <forward_list>
#include <list>

#include "test_range.h"
#include "types.h"

constexpr bool test() {
  std::list list                 = {1, 1, 1, 2, 2, 2, 3, 3};
  std::forward_list forward_list = {1, 1, 1, 2, 2, 2, 3, 3};

  // Test `chunk_view.end()`
  {
    auto view = list | std::views::chunk(3);
    auto it   = view.end();
    assert(std::ranges::equal(*--it, std::list{3, 3})); // We can adjust the tailing chunk-size.
    assert(std::ranges::equal(*--it, std::list{2, 2, 2}));
    assert(std::ranges::equal(*--it, std::list{1, 1, 1}));
  }

  // Test `not_sized_chunk_view.end()`
  {
    auto not_sized_list = not_sized_view(list | std::views::all);
    auto view           = not_sized_list | std::views::chunk(4);
    static_assert(std::ranges::bidirectional_range<decltype(view)>);
    static_assert(!std::ranges::sized_range<decltype(view)>);
    static_assert(
        std::same_as<
            decltype(view.end()),
            std::
                default_sentinel_t>); // We cannot handle the tailing chunk without size info, so we forbids one to derement from end().
  }

  // Test `forward_chunk_view.end()`
  {
    auto view = list | std::views::chunk(5);
    assert(++(++view.begin()) == view.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}