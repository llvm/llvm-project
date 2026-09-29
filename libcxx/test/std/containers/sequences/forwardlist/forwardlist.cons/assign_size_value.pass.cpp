//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void assign(size_type n, const value_type& v); // constexpr since C++26

#include <forward_list>
#include <cassert>
#include <iterator>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    typedef int T;
    typedef std::forward_list<T> C;
    const T t1[] = {10, 11, 12, 13};
    C c(std::begin(t1), std::end(t1));
    c.assign(10, 1);
    int n = 0;
    for (C::const_iterator i = c.cbegin(); i != c.cend(); ++i, ++n)
      assert(*i == 1);
    assert(n == 10);
  }
  {
    typedef int T;
    typedef std::forward_list<T> C;
    const T t1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    C c(std::begin(t1), std::end(t1));
    c.assign(4, 10);
    int n = 0;
    for (C::const_iterator i = c.cbegin(); i != c.cend(); ++i, ++n)
      assert(*i == 10);
    assert(n == 4);
  }
#if TEST_STD_VER >= 11
  {
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    const T t1[] = {10, 11, 12, 13};
    C c(std::begin(t1), std::end(t1));
    c.assign(10, 1);
    int n = 0;
    for (C::const_iterator i = c.cbegin(); i != c.cend(); ++i, ++n)
      assert(*i == 1);
    assert(n == 10);
  }
  {
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    const T t1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    C c(std::begin(t1), std::end(t1));
    c.assign(4, 10);
    int n = 0;
    for (C::const_iterator i = c.cbegin(); i != c.cend(); ++i, ++n)
      assert(*i == 10);
    assert(n == 4);
  }
#endif

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
