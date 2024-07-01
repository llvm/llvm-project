//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <generator>

// template<class Ref, class V = void, class Allocator = void>
//   class generator;

#include <generator>

#include <cassert>
#include <ranges>
#include <utility>
#include <vector>

std::generator<int> fib() {
  int a = 0;
  int b = 1;
  while (true) {
    co_yield std::exchange(a, std::exchange(b, a + b));
  }
}

std::generator<const int&> range_fib() {
  co_yield std::ranges::elements_of(std::vector<int>{0, 1});
  co_yield std::ranges::elements_of(std::vector<int>{1, 2});
  co_yield std::ranges::elements_of(std::vector<int>{3, 5});
  co_yield std::ranges::elements_of(std::vector<int>{5, 8});
}

bool test() {
  {
    std::vector<int> expected_fib_vec = {0, 1, 1, 2, 3};
    {
      auto fib_vec = fib() | std::views::take(5) | std::ranges::to<std::vector<int>>();
      assert(fib_vec == expected_fib_vec);
    }
    {
      auto fib_vec = range_fib() | std::views::take(5) | std::ranges::to<std::vector<int>>();
      assert(fib_vec == expected_fib_vec);
    }
  }
  return true;
}

int main() {
  test();
  return 0;
}
