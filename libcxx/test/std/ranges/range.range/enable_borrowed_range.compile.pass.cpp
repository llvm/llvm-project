//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// <ranges>

// template<class>
//  inline constexpr bool enable_borrowed_range = false;

#include <ranges>
#include <array>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "test_macros.h"

struct S {};

void test() {
  using std::ranges::enable_borrowed_range;
  static_assert(!enable_borrowed_range<char>);
  static_assert(!enable_borrowed_range<int>);
  static_assert(!enable_borrowed_range<double>);
  static_assert(!enable_borrowed_range<S>);

  // Sequence containers
  static_assert(!enable_borrowed_range<std::array<int, 0> >);
  static_assert(!enable_borrowed_range<std::array<int, 42> >);
  static_assert(!enable_borrowed_range<std::deque<int> >);
  static_assert(!enable_borrowed_range<std::forward_list<int> >);
  static_assert(!enable_borrowed_range<std::list<int> >);
  static_assert(!enable_borrowed_range<std::vector<int> >);

  // Associative containers
  static_assert(!enable_borrowed_range<std::set<int> >);
  static_assert(!enable_borrowed_range<std::map<int, int> >);
  static_assert(!enable_borrowed_range<std::multiset<int> >);
  static_assert(!enable_borrowed_range<std::multimap<int, int> >);

  // Unordered associative containers
  static_assert(!enable_borrowed_range<std::unordered_set<int> >);
  static_assert(!enable_borrowed_range<std::unordered_map<int, int> >);
  static_assert(!enable_borrowed_range<std::unordered_multiset<int> >);
  static_assert(!enable_borrowed_range<std::unordered_multimap<int, int> >);

  // Container adaptors
  static_assert(!enable_borrowed_range<std::stack<int> >);
  static_assert(!enable_borrowed_range<std::queue<int> >);
  static_assert(!enable_borrowed_range<std::priority_queue<int> >);

  // Both std::span and std::string_view have their own test.
}
