//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <set>

// class multiset

// template <class... Args>
//   iterator emplace(Args&&... args);

#include <cassert>
#include <set>

#include "../../Emplaceable.h"
#include "DefaultOnly.h"
#include "MoveOnly.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef std::multiset<DefaultOnly> M;
    typedef M::iterator R;
    M m;
    assert(DefaultOnly::count == 0);
    R r = m.emplace();
    assert(r == m.begin());
    assert(m.size() == 1);
    assert(*m.begin() == DefaultOnly());
    assert(DefaultOnly::count == 1);

    r = m.emplace();
    assert(r == std::next(m.begin()));
    assert(m.size() == 2);
    assert(*m.begin() == DefaultOnly());
    assert(DefaultOnly::count == 2);
  }
  assert(DefaultOnly::count == 0);
  {
    typedef std::multiset<Emplaceable> M;
    typedef M::iterator R;
    M m;
    R r = m.emplace();
    assert(r == m.begin());
    assert(m.size() == 1);
    assert(*m.begin() == Emplaceable());
    r = m.emplace(2, 3.5);
    assert(r == std::next(m.begin()));
    assert(m.size() == 2);
    assert(*r == Emplaceable(2, 3.5));
    r = m.emplace(2, 3.5);
    assert(r == std::next(m.begin(), 2));
    assert(m.size() == 3);
    assert(*r == Emplaceable(2, 3.5));
  }
  {
    typedef std::multiset<int> M;
    typedef M::iterator R;
    M m;
    R r = m.emplace(M::value_type(2));
    assert(r == m.begin());
    assert(m.size() == 1);
    assert(*r == 2);
  }
  {
    typedef std::multiset<int, std::less<int>, min_allocator<int>> M;
    typedef M::iterator R;
    M m;
    R r = m.emplace(M::value_type(2));
    assert(r == m.begin());
    assert(m.size() == 1);
    assert(*r == 2);
  }
  { // We're unwrapping pairs for `{,multi}map`. Make sure we're not trying to do that for multiset.
    using Set = std::multiset<std::pair<MoveOnly, MoveOnly>>;
    Set set;
    auto iter = set.emplace(std::pair<MoveOnly, MoveOnly>(2, 4));
    assert(set.begin() == iter);
  }

  return 0;
}
