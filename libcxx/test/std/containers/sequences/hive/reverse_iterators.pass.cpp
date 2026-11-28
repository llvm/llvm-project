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

// reverse_iterator       rbegin() noexcept;
// const_reverse_iterator rbegin() const noexcept;
// reverse_iterator       rend() noexcept;
// const_reverse_iterator rend() const noexcept;
// const_reverse_iterator crbegin() const noexcept;
// const_reverse_iterator crend() const noexcept;

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
  const H& ch                                                         = h;
  std::same_as<typename H::reverse_iterator> decltype(auto) b         = h.rbegin();
  std::same_as<typename H::reverse_iterator> decltype(auto) e         = h.rend();
  std::same_as<typename H::const_reverse_iterator> decltype(auto) cb  = ch.rbegin();
  std::same_as<typename H::const_reverse_iterator> decltype(auto) ce  = ch.rend();
  std::same_as<typename H::const_reverse_iterator> decltype(auto) cb2 = h.crbegin();
  std::same_as<typename H::const_reverse_iterator> decltype(auto) ce2 = h.crend();
  static_assert(noexcept(h.rbegin()));
  static_assert(noexcept(ch.rbegin()));
  static_assert(noexcept(h.rend()));
  static_assert(noexcept(ch.rend()));
  static_assert(noexcept(h.crbegin()));
  static_assert(noexcept(h.crend()));

  assert(b == e);
  assert(cb == ce);
  assert(cb2 == ce2);
  assert(b == cb);
}

// The reverse walk is the forward walk backwards, over several blocks, in every const form.
template <class A>
void test_elements() {
  using H = std::hive<int, A>;
  H h(std::hive_limits(3, 3));
  for (int i = 0; i != 10; ++i) {
    h.insert(i);
  }
  const H& ch = h;

  std::vector<int> forward(h.begin(), h.end());
  std::vector<int> sorted = forward;
  std::sort(sorted.begin(), sorted.end());
  assert((sorted == std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));

  std::vector<int> backward(h.rbegin(), h.rend());
  std::reverse(backward.begin(), backward.end());
  assert(backward == forward);

  std::vector<int> cbackward(ch.rbegin(), ch.rend());
  std::reverse(cbackward.begin(), cbackward.end());
  assert(cbackward == forward);

  std::vector<int> cbackward2(h.crbegin(), h.crend());
  std::reverse(cbackward2.begin(), cbackward2.end());
  assert(cbackward2 == forward);

  assert(*h.rbegin() == *std::prev(h.end()));
  assert(*std::prev(h.rend()) == *h.begin());
  assert(h.rbegin().base() == h.end());
  assert(h.rend().base() == h.begin());
  check_invariant(h);
}

int main(int, char**) {
  test_types_and_noexcept<std::allocator<int>>();
  test_types_and_noexcept<min_allocator<int>>();
  test_elements<std::allocator<int>>();
  test_elements<min_allocator<int>>();

  return 0;
}
