//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// template <class InputIterator>
//     iterator insert_after(const_iterator p,
//                           InputIterator first, InputIterator last); // constexpr since C++26

#include <forward_list>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    typedef int T;
    typedef std::forward_list<T> C;
    typedef C::iterator I;
    typedef cpp17_input_iterator<const T*> J;
    C c;
    const T t[] = {0, 1, 2, 3, 4};
    I i         = c.insert_after(c.cbefore_begin(), J(t), J(t));
    assert(i == c.before_begin());
    assert(std::distance(c.begin(), c.end()) == 0);

    i = c.insert_after(c.cbefore_begin(), J(t), J(t + 3));
    assert(i == std::next(c.before_begin(), 3));
    assert(std::distance(c.begin(), c.end()) == 3);
    assert(*std::next(c.begin(), 0) == 0);
    assert(*std::next(c.begin(), 1) == 1);
    assert(*std::next(c.begin(), 2) == 2);

    i = c.insert_after(c.begin(), J(t + 3), J(t + 5));
    assert(i == std::next(c.begin(), 2));
    assert(std::distance(c.begin(), c.end()) == 5);
    assert(*std::next(c.begin(), 0) == 0);
    assert(*std::next(c.begin(), 1) == 3);
    assert(*std::next(c.begin(), 2) == 4);
    assert(*std::next(c.begin(), 3) == 1);
    assert(*std::next(c.begin(), 4) == 2);
  }
#if TEST_STD_VER >= 11
  {
    typedef int T;
    typedef std::forward_list<T, min_allocator<T>> C;
    typedef C::iterator I;
    typedef cpp17_input_iterator<const T*> J;
    C c;
    const T t[] = {0, 1, 2, 3, 4};
    I i         = c.insert_after(c.cbefore_begin(), J(t), J(t));
    assert(i == c.before_begin());
    assert(std::distance(c.begin(), c.end()) == 0);

    i = c.insert_after(c.cbefore_begin(), J(t), J(t + 3));
    assert(i == std::next(c.before_begin(), 3));
    assert(std::distance(c.begin(), c.end()) == 3);
    assert(*std::next(c.begin(), 0) == 0);
    assert(*std::next(c.begin(), 1) == 1);
    assert(*std::next(c.begin(), 2) == 2);

    i = c.insert_after(c.begin(), J(t + 3), J(t + 5));
    assert(i == std::next(c.begin(), 2));
    assert(std::distance(c.begin(), c.end()) == 5);
    assert(*std::next(c.begin(), 0) == 0);
    assert(*std::next(c.begin(), 1) == 3);
    assert(*std::next(c.begin(), 2) == 4);
    assert(*std::next(c.begin(), 3) == 1);
    assert(*std::next(c.begin(), 4) == 2);
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
