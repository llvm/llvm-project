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

// __hive_storage::__clear(): destroys every element and keeps every block, reserved, so the capacity stays.

#include <cassert>
#include <hive>
#include <memory>

#include "Counter.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class T, class A>
using Storage = std::__hive_storage<T, A>;

// Every element is destroyed, the size is 0, the capacity stays; a second clear is a no-op.
template <class A>
void test_destroys_and_keeps_blocks() {
  Storage<Counter<int>, A> s(std::hive_limits(3, 3), A());
  for (int n = 0; n != 7; ++n) {
    s.__emplace(n);
  }
  assert(Counter_base::gConstructed == 7);
  assert(s.__capacity() == 9);

  s.__clear();
  assert(Counter_base::gConstructed == 0);
  assert(s.__size() == 0);
  assert(s.__capacity() == 9);
  assert(s.__invariants());

  s.__clear();
  assert(Counter_base::gConstructed == 0);
  assert(s.__size() == 0);
  assert(s.__capacity() == 9);
  assert(s.__invariants());
}

// Empty storage: nothing to do.
void test_empty() {
  Storage<int, std::allocator<int>> s(std::allocator<int>{});
  s.__clear();
  assert(s.__size() == 0);
  assert(s.__capacity() == 0);
  assert(s.__invariants());
}

// An allocator with its own destroy gets one call per element, even for a trivially destructible type; the blocks
// are kept until the destructor.
void test_allocator_destroy() {
  test_allocator_statistics stats;
  {
    Storage<int, test_allocator<int>> s(std::hive_limits(3, 3), test_allocator<int>(&stats));
    for (int n = 0; n != 4; ++n) {
      s.__emplace(n);
    }
    assert(stats.alloc_count == 4);

    s.__clear();
    assert(stats.destroy_count == 4);
    assert(stats.alloc_count == 4);
  }
  assert(stats.alloc_count == 0);
}

int main(int, char**) {
  test_destroys_and_keeps_blocks<std::allocator<Counter<int>>>();
  test_destroys_and_keeps_blocks<min_allocator<Counter<int>>>();
  test_empty();
  test_allocator_destroy();

  return 0;
}
