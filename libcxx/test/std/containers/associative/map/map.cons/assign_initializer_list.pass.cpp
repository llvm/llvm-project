//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <map>

// class map

// map& operator=(initializer_list<value_type> il);

#include <map>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "test_allocator.h"

void test_basic() {
  {
    typedef std::pair<const int, double> V;
    std::map<int, double> m =
                            {
                                {20, 1},
                            };
    m =
                            {
                                {1, 1},
                                {1, 1.5},
                                {1, 2},
                                {2, 1},
                                {2, 1.5},
                                {2, 2},
                                {3, 1},
                                {3, 1.5},
                                {3, 2}
                            };
    assert(m.size() == 3);
    assert(distance(m.begin(), m.end()) == 3);
    assert(*m.begin() == V(1, 1));
    assert(*next(m.begin()) == V(2, 1));
    assert(*next(m.begin(), 2) == V(3, 1));
    }
    {
    typedef std::pair<const int, double> V;
    std::map<int, double, std::less<int>, min_allocator<V>> m =
                            {
                                {20, 1},
                            };
    m =
                            {
                                {1, 1},
                                {1, 1.5},
                                {1, 2},
                                {2, 1},
                                {2, 1.5},
                                {2, 2},
                                {3, 1},
                                {3, 1.5},
                                {3, 2}
                            };
    assert(m.size() == 3);
    assert(distance(m.begin(), m.end()) == 3);
    assert(*m.begin() == V(1, 1));
    assert(*next(m.begin()) == V(2, 1));
    assert(*next(m.begin(), 2) == V(3, 1));
    }
}


void duplicate_keys_test() {
  typedef std::map<int, int, std::less<int>, test_allocator<std::pair<const int, int> > > Map;
  {
    LIBCPP_ASSERT(test_alloc_base::alloc_count == 0);
    Map s = {{1, 0}, {2, 0}, {3, 0}};
    LIBCPP_ASSERT(test_alloc_base::alloc_count == 3);
    s = {{4, 0}, {4, 0}, {4, 0}, {4, 0}};
    LIBCPP_ASSERT(test_alloc_base::alloc_count == 1);
    assert(s.size() == 1);
    assert(s.begin()->first == 4);
  }
  LIBCPP_ASSERT(test_alloc_base::alloc_count == 0);
}

int main(int, char**)
{
  test_basic();
  duplicate_keys_test();

  return 0;
}
