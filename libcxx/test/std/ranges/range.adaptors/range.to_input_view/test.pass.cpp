//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

#include "assert_macros.h"
#include <iterator>
#include <memory>
#include <ranges>
#include <vector>

int main() {
  {
    auto input_view = std::vector<int>{} | std::ranges::views::to_input;
    TEST_REQUIRE(input_view.begin() == input_view.end(), "begin() == end() is false");
  }

  {
    auto input_view = std::vector<int>{100, 200} | std::ranges::views::to_input;
    auto it         = input_view.begin();
    TEST_REQUIRE(*it == 100, "*it == 100 is false");
    ++it;
    TEST_REQUIRE(*it == 200, "*it == 200 is false");
    ++it;
    TEST_REQUIRE(it == input_view.end(), "it == input_view.end() is false");
  }

  {
    auto input_view = std::vector<int>{1, 2, 3} | std::ranges::views::to_input;
    auto base_vec   = input_view.base();
    TEST_REQUIRE(base_vec.size() == 3, "base_vec.size() == 0 is false");
    TEST_REQUIRE(base_vec[0] == 1, "base_vec[0] == 1 is false");
  }

  {
    const std::vector<int> vec(5);
    auto input_view = std::ranges::views::to_input(vec);
    TEST_REQUIRE(input_view.size() == 5, "input_view.size() == 5 is false");
  }

  {
    std::vector<std::unique_ptr<int>> vec;
    vec.push_back(std::make_unique<int>(10));
    vec.push_back(std::make_unique<int>(20));

    auto input_view = std::ranges::views::to_input(std::move(vec));

    auto it   = input_view.begin();
    auto val1 = std::ranges::iter_move(it);
    TEST_REQUIRE(*val1 == 10, "*val1 == 10 is false");
    TEST_REQUIRE(it->get() == nullptr, "it->get() == nullptr is false");

    ++it;
    auto val2 = std::ranges::iter_move(it);
    TEST_REQUIRE(*val2 == 20, "*val2 == 20 is false");
    TEST_REQUIRE(it->get() == nullptr, "it->get() == nullptr is false");
  }

  {
    std::vector<int> vec{1, 2};
    auto input_view = std::ranges::views::to_input(vec);
    auto it1        = input_view.begin();
    auto it2        = std::next(it1);

    std::ranges::iter_swap(it1, it2);

    TEST_REQUIRE(vec[0] == 2, "vec[0] == 2 is false");
    TEST_REQUIRE(vec[1] == 1, "vec[1] == 1 is false");
  }
}