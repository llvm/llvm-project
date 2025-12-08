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
#include <list>

#include "test_range.h"

constexpr bool test() {
  std::list<int> full_list  = {1, 1, 1, 2, 2, 2, 3, 3};
  std::list<int> empty_list = {};

  // Test `chunk_view.begin()`
  {
    auto view = full_list | std::views::chunk(3);
    auto it   = view.begin();
    assert(std::ranges::equal(*it, std::list{1, 1, 1}));
    assert(std::ranges::equal(*++it, std::list{2, 2, 2}));
    assert(std::ranges::equal(*++it, std::list{3, 3})); // The last chunk has only 2 elements.
    assert(++it == view.end());                         // Reaches end

    view = full_list | std::views::chunk(5);
    it   = view.begin();
    assert(std::ranges::equal(*it, std::list{1, 1, 1, 2, 2}));
    assert(std::ranges::equal(*++it, std::list{2, 3, 3}));
  }

  // Test `empty_chunk_view.begin()`
  {
    auto view = empty_list | std::views::chunk(3);
    assert(view.size() == 0);
    assert(view.begin() == view.end());
  }

  // Test `small_view_with_big_chunk.begin()`
  {
    auto view = full_list | std::views::chunk(314159);
    assert(view.size() == 1);
    assert(std::ranges::equal(*view.begin(), full_list));
    assert(++view.begin() == view.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}