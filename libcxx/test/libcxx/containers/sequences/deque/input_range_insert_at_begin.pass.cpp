//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that inserting an unsized input range at the start of a deque uses
// existing front spare capacity instead of materializing the range first.

// UNSUPPORTED: c++03

#include <cassert>
#include <deque>

#include "count_new.h"
#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER >= 23
#  include <ranges>
#endif

std::deque<int> make_deque_with_front_spare() {
  std::deque<int> c;
  for (int i = 0; i != 8; ++i)
    c.push_back(100 + i);
  for (int i = 0; i != 4; ++i)
    c.pop_front();
  return c;
}

void assert_expected(const std::deque<int>& c) {
  int expected[] = {0, 1, 2, 3, 104, 105, 106, 107};
  assert(c.size() == 8);
  for (int i = 0; i != 8; ++i)
    assert(c[i] == expected[i]);
}

void test_insert_iter_iter() {
  std::deque<int> c = make_deque_with_front_spare();
  int input[]       = {0, 1, 2, 3};
  typedef cpp17_input_iterator<int*> Iter;

  {
    DisableAllocationGuard guard;
    auto it = c.insert(c.begin(), Iter(input), Iter(input + 4));
    assert(it == c.begin());
  }

  assert_expected(c);
}

#if TEST_STD_VER >= 23
void test_insert_range() {
  std::deque<int> c = make_deque_with_front_spare();
  int input[]       = {0, 1, 2, 3};
  using Iter        = cpp20_input_iterator<int*>;
  auto in           = std::ranges::subrange(Iter(input), sentinel_wrapper<Iter>(Iter(input + 4)));

  {
    DisableAllocationGuard guard;
    auto it = c.insert_range(c.begin(), in);
    assert(it == c.begin());
  }

  assert_expected(c);
}

void test_prepend_range() {
  std::deque<int> c = make_deque_with_front_spare();
  int input[]       = {0, 1, 2, 3};
  using Iter        = cpp20_input_iterator<int*>;
  auto in           = std::ranges::subrange(Iter(input), sentinel_wrapper<Iter>(Iter(input + 4)));

  {
    DisableAllocationGuard guard;
    c.prepend_range(in);
  }

  assert_expected(c);
}
#endif

int main(int, char**) {
  test_insert_iter_iter();
#if TEST_STD_VER >= 23
  test_insert_range();
  test_prepend_range();
#endif

  return 0;
}
