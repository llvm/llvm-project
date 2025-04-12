//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// explicit deque(const allocator_type& a);

#include "asan_testing.h"
#include <deque>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "../../../NotConstructible.h"
#include "min_allocator.h"

template <class T, class Allocator>
void test_util(const Allocator& a) {
  std::deque<T, Allocator> d(a);
  assert(d.size() == 0);
  assert(d.get_allocator() == a);
  LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(d));
}

TEST_CONSTEXPR_CXX26 bool test() {
  test_util<int>(std::allocator<int>());
  test_util<NotConstructible>(test_allocator<NotConstructible>(3));
#if TEST_STD_VER >= 11
  test_util<int>(min_allocator<int>());
  test_util<int>(safe_allocator<int>());
  test_util<NotConstructible>(min_allocator<NotConstructible>{});
  test_util<NotConstructible>(safe_allocator<NotConstructible>{});
  test_util<int>(explicit_allocator<int>());
  test_util<NotConstructible>(explicit_allocator<NotConstructible>{});
#endif
  return false;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
