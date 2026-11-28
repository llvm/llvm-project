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

// void clear() noexcept;

#include <cassert>
#include <hive>
#include <memory>

#include "../helpers.h"
#include "Counter.h"
#include "min_allocator.h"
#include "test_macros.h"

// Every element is destroyed, the hive is empty; the blocks are kept (libc++), so inserting again allocates nothing.
template <class A>
void test_clear() {
  using H = std::hive<Counter<int>, A>;
  H h(std::hive_limits(3, 3));
  for (int i = 0; i != 7; ++i) {
    h.insert(Counter<int>(i));
  }
  assert(Counter_base::gConstructed == 7);
  const auto capacity = h.capacity();
  static_assert(noexcept(h.clear()));

  h.clear();
  assert(Counter_base::gConstructed == 0);
  assert(h.empty());
  assert(h.size() == 0);
  assert(h.begin() == h.end());
  LIBCPP_ASSERT(h.capacity() == capacity);
  check_invariant(h);

  h.clear();
  assert(h.empty());
  LIBCPP_ASSERT(h.capacity() == capacity);
  check_invariant(h);

  h.insert(Counter<int>(1));
  assert(h.size() == 1);
  assert(Counter_base::gConstructed == 1);
  LIBCPP_ASSERT(h.capacity() == capacity);
  check_invariant(h);
}

// An empty hive stays empty.
template <class A>
void test_empty() {
  std::hive<int, A> h;
  h.clear();
  assert(h.empty());
  assert(h.capacity() == 0);
  check_invariant(h);
}

int main(int, char**) {
  test_clear<std::allocator<Counter<int>>>();
  assert(Counter_base::gConstructed == 0);
  test_clear<min_allocator<Counter<int>>>();
  assert(Counter_base::gConstructed == 0);
  test_empty<std::allocator<int>>();
  test_empty<min_allocator<int>>();

  return 0;
}
