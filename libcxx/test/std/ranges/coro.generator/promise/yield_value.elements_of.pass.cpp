//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <generator>
// yield_value(ranges::elements_of<R, Alloc>);

#include <generator>

#include <algorithm>
#include <cassert>
#include <iterator>
#include <ranges>
#include <span>
#include <vector>

#include "test_iterators.h"

std::generator<const int&> ranges() {
  int values[] = {1, 2, 3};
  using I      = cpp20_input_iterator<int*>;
  using S      = sentinel_wrapper<I>;
  auto range   = std::ranges::subrange(I(values), S(I(values + 3)));
  co_yield std::ranges::elements_of(range);                  // move-only input iterator, distinct sentinel
  co_yield std::ranges::elements_of(std::vector<int>{4, 5}); // temporary range lifetime
  co_yield std::ranges::elements_of(std::views::iota(6, 8)); // prvalue reference
  co_yield std::ranges::elements_of(std::span<int>{});       // empty range
}

// LWG 4119: the source range's value type need not be a valid value type for the nested generator.
std::generator<std::span<int>> spans(std::vector<std::vector<int>>& values) {
  co_yield std::ranges::elements_of(values);
}

std::generator<int> integers() {
  co_yield 8;
  co_yield 9;
}

std::generator<long> converted_generator() {
  co_yield std::ranges::elements_of(integers()); // different yielded types use the general range overload
}

int main(int, char**) {
  const int expected[] = {1, 2, 3, 4, 5, 6, 7};
  assert(std::ranges::equal(ranges(), expected));
  const long converted[] = {8, 9};
  assert(std::ranges::equal(converted_generator(), converted));

  std::vector<std::vector<int>> values{{1, 2}, {}, {3}};
  auto g = spans(values);
  auto i = g.begin();
  assert((*i).data() == values[0].data());
  assert((*i).size() == 2);
  (*i)[0] = 42;
  assert(values[0][0] == 42);
  ++i;
  assert((*i).empty());
  ++i;
  assert((*i).data() == values[2].data());
  assert((*i)[0] == 3);
  ++i;
  assert(i == g.end());
  return 0;
}
