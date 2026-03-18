//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Make sure that lookup methods on ordered associative containers work properly
// with comparators that require an implicit conversion on lookup. Using a
// comparator like less<S> with reference_wrapper<S> as the key type causes an
// implicit conversion from reference_wrapper<S> to const S& during comparison.
//
// Potential heterogeneous lookup optimizations must not break this by making the
// comparator "transparent" when doing so would remove that conversion.
//
// This is a regression test for https://llvm.org/PR179319.

#include <cassert>
#include <functional>
#include <map>
#include <set>

#include "test_macros.h"

struct S {
  int i_;

  S(int i) : i_(i) {}
  bool operator<(S lhs) const { return lhs.i_ < i_; }
};

template <class Container>
void test(Container& container) {
  // non-const
  {
    Container& c = container;
    S v(1);
    assert(c.find(v) == c.end());
    assert(c.count(v) == 0);
    assert(c.lower_bound(v) == c.end());
    assert(c.upper_bound(v) == c.end());
    assert(c.equal_range(v).first == c.end());
    assert(c.equal_range(v).second == c.end());
#if TEST_STD_VER >= 20
    assert(!c.contains(v));
#endif
  }

  // const
  {
    Container const& c = container;
    S v(1);
    assert(c.find(v) == c.end());
    assert(c.count(v) == 0);
    assert(c.lower_bound(v) == c.end());
    assert(c.upper_bound(v) == c.end());
    assert(c.equal_range(v).first == c.end());
    assert(c.equal_range(v).second == c.end());
#if TEST_STD_VER >= 20
    assert(!c.contains(v));
#endif
  }
}

int main(int, char**) {
  {
    std::map<std::reference_wrapper<S>, void*, std::less<S>> m;
    test(m);
  }
  {
    std::multimap<std::reference_wrapper<S>, void*, std::less<S>> m;
    test(m);
  }
  {
    std::set<std::reference_wrapper<S>, std::less<S>> s;
    test(s);
  }
  {
    std::multiset<std::reference_wrapper<S>, std::less<S>> s;
    test(s);
  }

  return 0;
}
