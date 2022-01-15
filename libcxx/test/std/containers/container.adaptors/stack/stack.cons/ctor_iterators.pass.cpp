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
// stack(InputIterator, InputIterator);

#include <cassert>
#include <stack>

#include "test_allocator.h"

static_assert(!std::is_constructible_v<std::stack<int>, int, int, std::allocator<int>>);
static_assert(!std::is_constructible_v<std::stack<int>, int*, int*, int>);
static_assert( std::is_constructible_v<std::stack<int, std::deque<int, test_allocator<int>>>, int*, int*, test_allocator<int>>);
static_assert(!std::is_constructible_v<std::stack<int, std::deque<int, test_allocator<int>>>, int*, int*, std::allocator<int>>);

struct alloc : test_allocator<int> {
  alloc(test_allocator_statistics* a);
};
static_assert( std::is_constructible_v<std::stack<int, std::deque<int, alloc>>, int*, int*, test_allocator_statistics*>);

int main(int, char**) {
  const int a[] = {4, 3, 2, 1};
  std::stack<int> stack(a, a + 4);
  assert(stack.top() == 1);
  stack.pop();
  assert(stack.top() == 2);
  stack.pop();
  assert(stack.top() == 3);
  stack.pop();
  assert(stack.top() == 4);
  stack.pop();
  assert(stack.empty());

  return 0;
}
