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

// __hive_storage::__emplace(args...): constructs in the free-list head, else in a reserved block, else in a new
// block of __next_capacity(), and returns the block and the slot.

#include <cassert>
#include <cstddef>
#include <hive>
#include <memory>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class T, class A>
using Storage = std::__hive_storage<T, A>;

struct Pair {
  int a;
  int b;
  Pair(int x, int y) : a(x), b(y) {}
};

// The first element: a new block of the minimum capacity, slot 0, arguments forwarded.
template <class A>
void test_first_element() {
  Storage<Pair, A> s(std::hive_limits(3, 3), A());
  auto [b, i] = s.__emplace(1, 2);
  assert(s.__invariants());
  assert(i == 0);
  assert(b->__capacity_ == 3);
  assert(b->__size_ == 1);
  assert(b->__number_ == 0);
  assert((*(b->__slots_ + i)).a == 1);
  assert((*(b->__slots_ + i)).b == 2);
  assert(s.__size() == 1);
  assert(s.__capacity() == 3);
}

// Slots fill in order; a full block leaves the free list, and the next element opens a block numbered after it.
template <class A>
void test_fills_then_grows() {
  Storage<int, A> s(std::hive_limits(3, 3), A());
  auto [b0, i0] = s.__emplace(10);
  auto [b1, i1] = s.__emplace(11);
  auto [b2, i2] = s.__emplace(12);
  assert(b0 == b1);
  assert(b1 == b2);
  assert(i0 == 0);
  assert(i1 == 1);
  assert(i2 == 2);
  assert(s.__invariants());
  assert(b0->__size_ == 3);
  assert(s.__size() == 3);
  assert(s.__capacity() == 3);

  auto [b3, i3] = s.__emplace(13);
  assert(b3 != b0);
  assert(i3 == 0);
  assert(b3->__number_ == b0->__number_ + 1);
  assert(b0->__next_ == b3);
  assert(b3->__prev_ == b0);
  assert(s.__invariants());
  assert(s.__size() == 4);
  assert(s.__capacity() == 6);
  assert(*(b0->__slots_ + 0) == 10);
  assert(*(b0->__slots_ + 1) == 11);
  assert(*(b0->__slots_ + 2) == 12);
  assert(*(b3->__slots_ + 0) == 13);
}

// Default limits for int, {64, 4096}: block capacities follow the size, rounded up to whole words.
void test_growth() {
  Storage<int, std::allocator<int>> s(std::allocator<int>{});
  const std::size_t expected[] = {64, 64, 128, 256, 512, 1024, 2048, 4096, 4096};
  std::size_t blocks           = 0;
  for (int n = 0; n != 10000; ++n) {
    auto [b, i] = s.__emplace(n);
    if (i == 0) {
      assert(blocks < 9);
      assert(b->__capacity_ == expected[blocks]);
      ++blocks;
    }
    assert(*(b->__slots_ + i) == n);
  }
  assert(blocks == 9);
  assert(s.__size() == 10000);
  assert(s.__capacity() == 12288);
  assert(s.__invariants());
}

// After __clear the blocks are reserved: __emplace takes them back, slot 0 first, before it allocates.
template <class A>
void test_reuses_reserved_blocks() {
  Storage<int, A> s(std::hive_limits(3, 3), A());
  for (int n = 0; n != 4; ++n) {
    s.__emplace(n);
  }
  s.__clear();
  assert(s.__size() == 0);
  assert(s.__capacity() == 6);

  auto [b, i] = s.__emplace(10);
  assert(i == 0);
  assert(b->__number_ == 0);
  assert(s.__size() == 1);
  assert(s.__capacity() == 6);
  assert(s.__invariants());

  s.__emplace(11);
  s.__emplace(12);
  auto [b2, i2] = s.__emplace(13);
  assert(b2 != b);
  assert(i2 == 0);
  assert(b2->__number_ == 1);
  assert(s.__capacity() == 6);
  assert(s.__invariants());

  s.__emplace(14);
  s.__emplace(15);
  auto [b3, i3] = s.__emplace(16);
  assert(b3 != b2);
  assert(i3 == 0);
  assert(s.__size() == 7);
  assert(s.__capacity() == 9);
  assert(s.__invariants());
}

// Two allocations per block and one construct per element, all through the allocator; nothing leaks.
void test_allocator_use() {
  test_allocator_statistics stats;
  {
    Storage<int, test_allocator<int>> s(std::hive_limits(2, 2), test_allocator<int>(&stats));
    s.__emplace(1);
    assert(stats.alloc_count == 2);
    assert(stats.construct_count == 1);
    s.__emplace(2);
    assert(stats.alloc_count == 2);
    assert(stats.construct_count == 2);
    s.__emplace(3);
    assert(stats.alloc_count == 4);
    assert(stats.construct_count == 3);
  }
  assert(stats.alloc_count == 0);
}

int main(int, char**) {
  test_first_element<std::allocator<Pair>>();
  test_first_element<min_allocator<Pair>>();
  test_fills_then_grows<std::allocator<int>>();
  test_fills_then_grows<min_allocator<int>>();
  test_growth();
  test_reuses_reserved_blocks<std::allocator<int>>();
  test_reuses_reserved_blocks<min_allocator<int>>();
  test_allocator_use();

  return 0;
}
