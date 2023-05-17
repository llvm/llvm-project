//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <queue>

// template <class InputIterator>
// queue(InputIterator, InputIterator);

#include <cassert>
#include <queue>
#include <type_traits>

#include "test_allocator.h"

static_assert(!std::is_constructible_v<std::queue<int>, int, int, std::allocator<int>>);
static_assert(!std::is_constructible_v<std::queue<int>, int*, int*, int>);
static_assert( std::is_constructible_v<std::queue<int, std::deque<int, test_allocator<int>>>, int*, int*, test_allocator<int>>);
static_assert(!std::is_constructible_v<std::queue<int, std::deque<int, test_allocator<int>>>, int*, int*, std::allocator<int>>);

template <class T>
struct alloc : test_allocator<T> {
  alloc(test_allocator_statistics* a);

  template <class U>
  struct rebind {
    using other = alloc<U>;
  };
};
static_assert(
    std::is_constructible_v<std::queue<int, std::deque<int, alloc<int>>>, int*, int*, test_allocator_statistics*>);

int main(int, char**) {
  const int a[] = {4, 3, 2, 1};
  std::queue<int> queue(a, a + 4);
  assert(queue.front() == 4);
  queue.pop();
  assert(queue.front() == 3);
  queue.pop();
  assert(queue.front() == 2);
  queue.pop();
  assert(queue.front() == 1);
  queue.pop();
  assert(queue.empty());

  return 0;
}
