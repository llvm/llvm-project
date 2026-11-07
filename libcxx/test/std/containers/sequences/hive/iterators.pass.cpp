//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive

// <hive>

// iterator       begin() noexcept;
// const_iterator begin() const noexcept;
// iterator       end() noexcept;
// const_iterator end() const noexcept;
// const_iterator cbegin() const noexcept;
// const_iterator cend() const noexcept;

#include <algorithm>
#include <cassert>
#include <concepts>
#include <hive>
#include <iterator>
#include <memory>
#include <vector>

#include "helpers.h"
#include "min_allocator.h"
#include "test_macros.h"

template <class A>
void test_types_and_noexcept() {
  using H = std::hive<int, A>;
  H h;
  const H& ch                                                 = h;
  std::same_as<typename H::iterator> decltype(auto) b         = h.begin();
  std::same_as<typename H::iterator> decltype(auto) e         = h.end();
  std::same_as<typename H::const_iterator> decltype(auto) cb  = ch.begin();
  std::same_as<typename H::const_iterator> decltype(auto) ce  = ch.end();
  std::same_as<typename H::const_iterator> decltype(auto) cb2 = h.cbegin();
  std::same_as<typename H::const_iterator> decltype(auto) ce2 = h.cend();
  static_assert(noexcept(h.begin()));
  static_assert(noexcept(ch.begin()));
  static_assert(noexcept(h.end()));
  static_assert(noexcept(ch.end()));
  static_assert(noexcept(h.cbegin()));
  static_assert(noexcept(h.cend()));

  // Empty: every pair is equal, across constness too.
  assert(b == e);
  assert(cb == ce);
  assert(cb2 == ce2);
  assert(b == cb);
  assert(cb == b);
}

// Three elements: the walk visits each once, the const walks see the same, a write through begin() is visible.
template <class A>
void test_elements() {
  using H = std::hive<int, A>;
  H h;
  h.insert(1);
  h.insert(2);
  h.insert(3);
  const H& ch = h;

  std::vector<int> seen(h.begin(), h.end());
  std::sort(seen.begin(), seen.end());
  assert((seen == std::vector<int>{1, 2, 3}));

  std::vector<int> cseen(ch.begin(), ch.end());
  std::sort(cseen.begin(), cseen.end());
  assert(cseen == seen);

  std::vector<int> cseen2(h.cbegin(), h.cend());
  std::sort(cseen2.begin(), cseen2.end());
  assert(cseen2 == seen);

  assert(h.begin() == ch.begin());
  assert(h.end() == ch.end());
  assert(std::next(h.begin(), 3) == h.end());
  assert(std::prev(h.end(), 3) == h.begin());

  const int first = *h.begin();
  *h.begin()      = 10;
  assert(*ch.begin() == 10);
  assert(std::count(h.begin(), h.end(), first) == 0);
  check_invariant(h);
}

// Enough elements for several blocks under small limits.
template <class A>
void test_across_blocks() {
  using H = std::hive<int, A>;
  H h(std::hive_limits(3, 3));
  std::vector<int> expected;
  for (int i = 0; i != 20; ++i) {
    h.insert(i);
    expected.push_back(i);
  }
  std::vector<int> seen(h.begin(), h.end());
  std::sort(seen.begin(), seen.end());
  assert(seen == expected);
  assert(std::distance(h.begin(), h.end()) == 20);
  assert(std::prev(std::next(h.begin(), 20), 20) == h.begin());
  check_invariant(h);
}

int main(int, char**) {
  test_types_and_noexcept<std::allocator<int>>();
  test_types_and_noexcept<min_allocator<int>>();
  test_elements<std::allocator<int>>();
  test_elements<min_allocator<int>>();
  test_across_blocks<std::allocator<int>>();
  test_across_blocks<min_allocator<int>>();

  return 0;
}
