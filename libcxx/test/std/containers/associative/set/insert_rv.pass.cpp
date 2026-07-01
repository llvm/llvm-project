//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <set>

// class set

// constexpr pair<iterator, bool> insert(value_type&& v); // constexpr since C++26

#include <set>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX26 bool test() {
  {
    typedef std::set<MoveOnly> M;
    typedef std::pair<M::iterator, bool> R;
    M m;
    R r = m.insert(M::value_type(2));
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 1);
    assert(*r.first == 2);

    r = m.insert(M::value_type(1));
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 2);
    assert(*r.first == 1);

    r = m.insert(M::value_type(3));
    assert(r.second);
    assert(r.first == std::prev(m.end()));
    assert(m.size() == 3);
    assert(*r.first == 3);

    r = m.insert(M::value_type(3));
    assert(!r.second);
    assert(r.first == std::prev(m.end()));
    assert(m.size() == 3);
    assert(*r.first == 3);
  }
  {
    typedef std::set<MoveOnly, std::less<MoveOnly>, min_allocator<MoveOnly>> M;
    typedef std::pair<M::iterator, bool> R;
    M m;
    R r = m.insert(M::value_type(2));
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 1);
    assert(*r.first == 2);

    r = m.insert(M::value_type(1));
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 2);
    assert(*r.first == 1);

    r = m.insert(M::value_type(3));
    assert(r.second);
    assert(r.first == std::prev(m.end()));
    assert(m.size() == 3);
    assert(*r.first == 3);

    r = m.insert(M::value_type(3));
    assert(!r.second);
    assert(r.first == std::prev(m.end()));
    assert(m.size() == 3);
    assert(*r.first == 3);
  }

  return true;
}
int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
