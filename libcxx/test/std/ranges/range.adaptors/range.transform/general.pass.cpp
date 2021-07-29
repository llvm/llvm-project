//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// Some basic examples of how transform_view might be used in the wild. This is a general
// collection of sample algorithms and functions that try to mock general usage of
// this view.

#include <ranges>

#include <cctype>
#include <functional>
#include <list>
#include <numeric>
#include <string>
#include <vector>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"
#include "types.h"

template<std::ranges::range R>
auto toUpper(R range) {
  return std::ranges::transform_view(range, [](char c) { return std::toupper(c); });
}

unsigned badRandom() { return 42; }

template<std::ranges::range R, class Fn = std::plus<std::iter_value_t<R>>>
auto withRandom(R&& range, Fn func = Fn()) {
  return std::ranges::transform_view(range, std::bind_front(func, badRandom()));
}

template<class E1, class E2, size_t N, class Join = std::plus<E1>>
auto joinArrays(E1 (&a)[N], E2 (&b)[N], Join join = Join()) {
  return std::ranges::transform_view(a, [&a, &b, join](auto& x) {
    auto idx = (&x) - a;
    return join(x, b[idx]);
  });
}

int main(int, char**) {
  {
    std::vector vec = {1, 2, 3, 4};
    auto sortOfRandom = withRandom(vec);
    std::vector check = {43, 44, 45, 46};
    assert(std::equal(sortOfRandom.begin(), sortOfRandom.end(), check.begin(), check.end()));
  }

  {
    int a[4] = {1, 2, 3, 4};
    int b[4] = {4, 3, 2, 1};
    auto out = joinArrays(a, b);
    int check[4] = {5, 5, 5, 5};
    assert(std::equal(out.begin(), out.end(), check));
  }

  {
    std::string_view str = "Hello, World.";
    auto upp = toUpper(str);
    std::string_view check = "HELLO, WORLD.";
    assert(std::equal(upp.begin(), upp.end(), check.begin(), check.end()));
  }

  return 0;
}
