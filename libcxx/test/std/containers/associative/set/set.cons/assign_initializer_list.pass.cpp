//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <set>

// class set

// set& operator=(initializer_list<value_type> il);

#include <set>
#include <cassert>
#include <iostream>

#include "test_macros.h"
#include "min_allocator.h"
#include "test_allocator.h"

void basic_test() {
  {
    typedef std::set<int> C;
    typedef C::value_type V;
    C m = {10, 8};
    m = {1, 2, 3, 4, 5, 6};
    assert(m.size() == 6);
    assert(distance(m.begin(), m.end()) == 6);
    C::const_iterator i = m.cbegin();
    assert(*i == V(1));
    assert(*++i == V(2));
    assert(*++i == V(3));
    assert(*++i == V(4));
    assert(*++i == V(5));
    assert(*++i == V(6));
  }
  {
    typedef std::set<int, std::less<int>, min_allocator<int> > C;
    typedef C::value_type V;
    C m = {10, 8};
    m = {1, 2, 3, 4, 5, 6};
    assert(m.size() == 6);
    assert(distance(m.begin(), m.end()) == 6);
    C::const_iterator i = m.cbegin();
    assert(*i == V(1));
    assert(*++i == V(2));
    assert(*++i == V(3));
    assert(*++i == V(4));
    assert(*++i == V(5));
    assert(*++i == V(6));
  }
}

void duplicate_keys_test() {
  typedef std::set<int, std::less<int>, test_allocator<int> > Set;
  {
    LIBCPP_ASSERT(test_alloc_base::alloc_count == 0);
    Set s = {1, 2, 3};
    LIBCPP_ASSERT(test_alloc_base::alloc_count == 3);
    s = {4, 4, 4, 4, 4};
    LIBCPP_ASSERT(test_alloc_base::alloc_count == 1);
    assert(s.size() == 1);
    assert(*s.begin() == 4);
  }
  LIBCPP_ASSERT(test_alloc_base::alloc_count == 0);
}

int main(int, char**) {
  basic_test();
  duplicate_keys_test();

  return 0;
}
