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

// template <class... Args>
//   pair<iterator, bool> emplace(Args&&... args);

#include <cassert>
#include <set>

#include "../../Emplaceable.h"
#include "DefaultOnly.h"
#include "MoveOnly.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef std::set<DefaultOnly> M;
    typedef std::pair<M::iterator, bool> R;
    M m;
    assert(DefaultOnly::count == 0);
    R r = m.emplace();
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 1);
    assert(*m.begin() == DefaultOnly());
    assert(DefaultOnly::count == 1);

    r = m.emplace();
    assert(!r.second);
    assert(r.first == m.begin());
    assert(m.size() == 1);
    assert(*m.begin() == DefaultOnly());
    assert(DefaultOnly::count == 1);
  }
  assert(DefaultOnly::count == 0);
  {
    typedef std::set<Emplaceable> M;
    typedef std::pair<M::iterator, bool> R;
    M m;
    R r = m.emplace();
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 1);
    assert(*m.begin() == Emplaceable());
    r = m.emplace(2, 3.5);
    assert(r.second);
    assert(r.first == std::next(m.begin()));
    assert(m.size() == 2);
    assert(*r.first == Emplaceable(2, 3.5));
    r = m.emplace(2, 3.5);
    assert(!r.second);
    assert(r.first == std::next(m.begin()));
    assert(m.size() == 2);
    assert(*r.first == Emplaceable(2, 3.5));
  }
  {
    typedef std::set<int> M;
    typedef std::pair<M::iterator, bool> R;
    M m;
    R r = m.emplace(M::value_type(2));
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 1);
    assert(*r.first == 2);
  }
  {
    typedef std::set<int, std::less<int>, min_allocator<int>> M;
    typedef std::pair<M::iterator, bool> R;
    M m;
    R r = m.emplace(M::value_type(2));
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 1);
    assert(*r.first == 2);
  }
  { // We're unwrapping pairs for `{,multi}map`. Make sure we're not trying to do that for set.
    using Set = std::set<std::pair<MoveOnly, MoveOnly>>;
    Set set;
    auto res = set.emplace(std::pair<MoveOnly, MoveOnly>(2, 4));
    assert(std::get<1>(res));
    assert(set.begin() == std::get<0>(res));
  }

  return 0;
}
