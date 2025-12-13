//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <forward_list>

// forward_list& operator=(forward_list&& x); // constexpr since C++26

#include <forward_list>
#include <cassert>
#include <iterator>

#include "test_macros.h"
#include "test_allocator.h"
#include "MoveOnly.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    typedef MoveOnly T;
    typedef test_allocator<T> A;
    typedef std::forward_list<T, A> C;
    T t0[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    T t1[] = {10, 11, 12, 13};
    typedef std::move_iterator<T*> I;
    C c0(I(std::begin(t0)), I(std::end(t0)), A(10));
    C c1(I(std::begin(t1)), I(std::end(t1)), A(10));
    c1    = std::move(c0);
    int n = 0;
    for (C::const_iterator i = c1.cbegin(); i != c1.cend(); ++i, (void)++n)
      assert(*i == n);
    assert(n == 10);
    assert(c1.get_allocator() == A(10));
    assert(c0.empty());
  }
  {
    typedef MoveOnly T;
    typedef test_allocator<T> A;
    typedef std::forward_list<T, A> C;
    T t0[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    T t1[] = {10, 11, 12, 13};
    typedef std::move_iterator<T*> I;
    C c0(I(std::begin(t0)), I(std::end(t0)), A(10));
    C c1(I(std::begin(t1)), I(std::end(t1)), A(11));
    c1    = std::move(c0);
    int n = 0;
    for (C::const_iterator i = c1.cbegin(); i != c1.cend(); ++i, (void)++n)
      assert(*i == n);
    assert(n == 10);
    assert(c1.get_allocator() == A(11));
    assert(!c0.empty());
  }
  {
    typedef MoveOnly T;
    typedef test_allocator<T> A;
    typedef std::forward_list<T, A> C;
    T t0[] = {10, 11, 12, 13};
    T t1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    typedef std::move_iterator<T*> I;
    C c0(I(std::begin(t0)), I(std::end(t0)), A(10));
    C c1(I(std::begin(t1)), I(std::end(t1)), A(10));
    c1    = std::move(c0);
    int n = 0;
    for (C::const_iterator i = c1.cbegin(); i != c1.cend(); ++i, (void)++n)
      assert(*i == 10 + n);
    assert(n == 4);
    assert(c1.get_allocator() == A(10));
    assert(c0.empty());
  }
  {
    typedef MoveOnly T;
    typedef test_allocator<T> A;
    typedef std::forward_list<T, A> C;
    T t0[] = {10, 11, 12, 13};
    T t1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    typedef std::move_iterator<T*> I;
    C c0(I(std::begin(t0)), I(std::end(t0)), A(10));
    C c1(I(std::begin(t1)), I(std::end(t1)), A(11));
    c1    = std::move(c0);
    int n = 0;
    for (C::const_iterator i = c1.cbegin(); i != c1.cend(); ++i, (void)++n)
      assert(*i == 10 + n);
    assert(n == 4);
    assert(c1.get_allocator() == A(11));
    assert(!c0.empty());
  }

  {
    typedef MoveOnly T;
    typedef other_allocator<T> A;
    typedef std::forward_list<T, A> C;
    T t0[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    T t1[] = {10, 11, 12, 13};
    typedef std::move_iterator<T*> I;
    C c0(I(std::begin(t0)), I(std::end(t0)), A(10));
    C c1(I(std::begin(t1)), I(std::end(t1)), A(10));
    c1    = std::move(c0);
    int n = 0;
    for (C::const_iterator i = c1.cbegin(); i != c1.cend(); ++i, (void)++n)
      assert(*i == n);
    assert(n == 10);
    assert(c1.get_allocator() == A(10));
    assert(c0.empty());
  }
  {
    typedef MoveOnly T;
    typedef other_allocator<T> A;
    typedef std::forward_list<T, A> C;
    T t0[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    T t1[] = {10, 11, 12, 13};
    typedef std::move_iterator<T*> I;
    C c0(I(std::begin(t0)), I(std::end(t0)), A(10));
    C c1(I(std::begin(t1)), I(std::end(t1)), A(11));
    c1    = std::move(c0);
    int n = 0;
    for (C::const_iterator i = c1.cbegin(); i != c1.cend(); ++i, (void)++n)
      assert(*i == n);
    assert(n == 10);
    assert(c1.get_allocator() == A(10));
    assert(c0.empty());
  }
  {
    typedef MoveOnly T;
    typedef other_allocator<T> A;
    typedef std::forward_list<T, A> C;
    T t0[] = {10, 11, 12, 13};
    T t1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    typedef std::move_iterator<T*> I;
    C c0(I(std::begin(t0)), I(std::end(t0)), A(10));
    C c1(I(std::begin(t1)), I(std::end(t1)), A(10));
    c1    = std::move(c0);
    int n = 0;
    for (C::const_iterator i = c1.cbegin(); i != c1.cend(); ++i, (void)++n)
      assert(*i == 10 + n);
    assert(n == 4);
    assert(c1.get_allocator() == A(10));
    assert(c0.empty());
  }
  {
    typedef MoveOnly T;
    typedef other_allocator<T> A;
    typedef std::forward_list<T, A> C;
    T t0[] = {10, 11, 12, 13};
    T t1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    typedef std::move_iterator<T*> I;
    C c0(I(std::begin(t0)), I(std::end(t0)), A(10));
    C c1(I(std::begin(t1)), I(std::end(t1)), A(11));
    c1    = std::move(c0);
    int n = 0;
    for (C::const_iterator i = c1.cbegin(); i != c1.cend(); ++i, (void)++n)
      assert(*i == 10 + n);
    assert(n == 4);
    assert(c1.get_allocator() == A(10));
    assert(c0.empty());
  }
  {
    typedef MoveOnly T;
    typedef min_allocator<T> A;
    typedef std::forward_list<T, A> C;
    T t0[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    T t1[] = {10, 11, 12, 13};
    typedef std::move_iterator<T*> I;
    C c0(I(std::begin(t0)), I(std::end(t0)), A());
    C c1(I(std::begin(t1)), I(std::end(t1)), A());
    c1    = std::move(c0);
    int n = 0;
    for (C::const_iterator i = c1.cbegin(); i != c1.cend(); ++i, (void)++n)
      assert(*i == n);
    assert(n == 10);
    assert(c1.get_allocator() == A());
    assert(c0.empty());
  }
  {
    typedef MoveOnly T;
    typedef min_allocator<T> A;
    typedef std::forward_list<T, A> C;
    T t0[] = {10, 11, 12, 13};
    T t1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    typedef std::move_iterator<T*> I;
    C c0(I(std::begin(t0)), I(std::end(t0)), A());
    C c1(I(std::begin(t1)), I(std::end(t1)), A());
    c1    = std::move(c0);
    int n = 0;
    for (C::const_iterator i = c1.cbegin(); i != c1.cend(); ++i, (void)++n)
      assert(*i == 10 + n);
    assert(n == 4);
    assert(c1.get_allocator() == A());
    assert(c0.empty());
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
