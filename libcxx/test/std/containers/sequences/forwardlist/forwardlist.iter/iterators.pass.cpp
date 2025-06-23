//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// iterator       begin();        // constexpr since C++26
// iterator       end();          // constexpr since C++26
// const_iterator begin()  const; // constexpr since C++26
// const_iterator end()    const; // constexpr since C++26
// const_iterator cbegin() const; // constexpr since C++26
// const_iterator cend()   const; // constexpr since C++26

#include <forward_list>
#include <cassert>
#include <iterator>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    typedef int T;
    typedef std::forward_list<T> C;
    C c;
    C::iterator i = c.begin();
    C::iterator j = c.end();
    assert(std::distance(i, j) == 0);
    assert(i == j);
  }
  {
    typedef int T;
    typedef std::forward_list<T> C;
    const C c;
    C::const_iterator i = c.begin();
    C::const_iterator j = c.end();
    assert(std::distance(i, j) == 0);
    assert(i == j);
  }
  {
    typedef int T;
    typedef std::forward_list<T> C;
    C c;
    C::const_iterator i = c.cbegin();
    C::const_iterator j = c.cend();
    assert(std::distance(i, j) == 0);
    assert(i == j);
    assert(i == c.end());
  }
  {
    typedef int T;
    typedef std::forward_list<T> C;
    const T t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    C c(std::begin(t), std::end(t));
    C::iterator i = c.begin();
    assert(*i == 0);
    ++i;
    assert(*i == 1);
    *i = 10;
    assert(*i == 10);
    assert(std::distance(c.begin(), c.end()) == 10);
  }
  {
    typedef int T;
    typedef std::forward_list<T> C;
    C::iterator i;
    C::const_iterator j;
    (void)i;
    (void)j;
  }
#if TEST_STD_VER >= 11
  {
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    C c;
    C::iterator i = c.begin();
    C::iterator j = c.end();
    assert(std::distance(i, j) == 0);
    assert(i == j);
  }
  {
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    const C c;
    C::const_iterator i = c.begin();
    C::const_iterator j = c.end();
    assert(std::distance(i, j) == 0);
    assert(i == j);
  }
  {
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    C c;
    C::const_iterator i = c.cbegin();
    C::const_iterator j = c.cend();
    assert(std::distance(i, j) == 0);
    assert(i == j);
    assert(i == c.end());
  }
  {
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    const T t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    C c(std::begin(t), std::end(t));
    C::iterator i = c.begin();
    assert(*i == 0);
    ++i;
    assert(*i == 1);
    *i = 10;
    assert(*i == 10);
    assert(std::distance(c.begin(), c.end()) == 10);
  }
  {
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    C::iterator i;
    C::const_iterator j;
    (void)i;
    (void)j;
  }
#endif
#if TEST_STD_VER > 11
  { // N3644 testing
    std::forward_list<int>::iterator ii1{}, ii2{};
    std::forward_list<int>::iterator ii4 = ii1;
    std::forward_list<int>::const_iterator cii{};
    assert(ii1 == ii2);
    assert(ii1 == ii4);

    assert(!(ii1 != ii2));

    assert((ii1 == cii));
    assert((cii == ii1));
    assert(!(ii1 != cii));
    assert(!(cii != ii1));

    //         std::forward_list<int> c;
    //         assert ( ii1 != c.cbegin());
    //         assert ( cii != c.begin());
    //         assert ( cii != c.cend());
    //         assert ( ii1 != c.end());
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
