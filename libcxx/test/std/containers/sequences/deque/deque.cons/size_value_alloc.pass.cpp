//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// constexpr since C++26

// deque(size_type n, const value_type& v, const allocator_type& a);

#include "asan_testing.h"
#include <deque>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "min_allocator.h"

template <class T, class Allocator>
void test(unsigned n, const T& x, const Allocator& a) {
  typedef std::deque<T, Allocator> C;
  typedef typename C::const_iterator const_iterator;
  C d(n, x, a);
  assert(d.get_allocator() == a);
  assert(d.size() == n);
  assert(static_cast<std::size_t>(std::distance(d.begin(), d.end())) == d.size());
  LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(d));
  for (const_iterator i = d.begin(), e = d.end(); i != e; ++i)
    assert(*i == x);
}

#if TEST_STD_VER >= 26
TEST_CONSTEXPR_CXX26 bool test_constexpr() {
  std::deque<int> d(3, 7, std::allocator<int>());
  assert(d.size() == 3);
  assert(d[1] == 7);
  return true;
}
#endif

int main(int, char**) {
#if TEST_STD_VER >= 26
  assert(test_constexpr());
  static_assert(test_constexpr());
#endif

  {
    std::allocator<int> a;
    test(0, 5, a);
    test(1, 10, a);
    test(10, 11, a);
    test(1023, -11, a);
    test(1024, 25, a);
    test(1025, 0, a);
    test(2047, 110, a);
    test(2048, -500, a);
    test(2049, 654, a);
    test(4095, 78, a);
    test(4096, 1165, a);
    test(4097, 157, a);
  }
#if TEST_STD_VER >= 11
  {
    min_allocator<int> a;
    test(0, 5, a);
    test(1, 10, a);
    test(10, 11, a);
    test(1023, -11, a);
    test(1024, 25, a);
    test(1025, 0, a);
    test(2047, 110, a);
    test(2048, -500, a);
    test(2049, 654, a);
    test(4095, 78, a);
    test(4096, 1165, a);
    test(4097, 157, a);
  }
#endif

  return 0;
}
