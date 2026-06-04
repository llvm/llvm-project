//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// constexpr since C++26

// deque()

#include "asan_testing.h"
#include <deque>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "../../../NotConstructible.h"
#include "min_allocator.h"

template <class T, class Allocator>
void test() {
  std::deque<T, Allocator> d;
  assert(d.size() == 0);
  LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(d));
#if TEST_STD_VER >= 11
  std::deque<T, Allocator> d1 = {};
  assert(d1.size() == 0);
  LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(d1));
#endif
}

#if TEST_STD_VER >= 26
constexpr bool test() {
  std::deque<int> d;
  assert(d.empty());
  assert(d.begin() == d.end());
  return true;
}
#endif

int main(int, char**) {
#if TEST_STD_VER >= 26
  test();
  static_assert(test());
#endif

  test<int, std::allocator<int> >();
  test<NotConstructible, limited_allocator<NotConstructible, 1> >();
#if TEST_STD_VER >= 11
  test<int, min_allocator<int> >();
  test<NotConstructible, min_allocator<NotConstructible> >();
#endif

  return 0;
}
