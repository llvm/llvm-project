//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <forward_list>

// forward_list& operator=(initializer_list<value_type> il); // constexpr since C++26

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
    c     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int n = 0;
    for (C::const_iterator i = c.cbegin(); i != c.cend(); ++i, ++n)
      assert(*i == n);
    assert(n == 10);
  }
  {
    typedef int T;
    typedef std::forward_list<T> C;
    const T t1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    C c(std::begin(t1), std::end(t1));
    c     = {10, 11, 12, 13};
    int n = 0;
    for (C::const_iterator i = c.cbegin(); i != c.cend(); ++i, ++n)
      assert(*i == 10 + n);
    assert(n == 4);
  }
  {
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    const T t1[] = {10, 11, 12, 13};
    C c(std::begin(t1), std::end(t1));
    c     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int n = 0;
    for (C::const_iterator i = c.cbegin(); i != c.cend(); ++i, ++n)
      assert(*i == n);
    assert(n == 10);
  }
  {
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    const T t1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    C c(std::begin(t1), std::end(t1));
    c     = {10, 11, 12, 13};
    int n = 0;
    for (C::const_iterator i = c.cbegin(); i != c.cend(); ++i, ++n)
      assert(*i == 10 + n);
    assert(n == 4);
  }

  return true;
}

int main(int, char**) {
  assert(test());
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
