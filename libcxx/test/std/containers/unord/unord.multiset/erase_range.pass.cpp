//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_multiset

// iterator erase(const_iterator first, const_iterator last)

#include <unordered_set>
#include <cassert>
#include <iterator>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef std::unordered_multiset<int> C;
    typedef int P;
    P a[] = {P(1), P(2), P(3), P(4), P(1), P(2)};
    C c(a, a + sizeof(a) / sizeof(a[0]));
    C::const_iterator i = c.find(2);
    C::const_iterator j = std::next(i, 2);
    C::iterator k       = c.erase(i, i);
    assert(k == i);
    assert(c.size() == 6);
    assert(c.count(1) == 2);
    assert(c.count(2) == 2);
    assert(c.count(3) == 1);
    assert(c.count(4) == 1);

    k = c.erase(i, j);
    assert(c.size() == 4);
    assert(c.count(1) == 2);
    assert(c.count(3) == 1);
    assert(c.count(4) == 1);

    k = c.erase(c.cbegin(), c.cend());
    assert(c.size() == 0);
    assert(k == c.end());
  }
  { // Make sure that we're properly updating the bucket list when starting within a bucket
    struct MyHash {
      size_t operator()(size_t v) const { return v; }
    };
    std::unordered_multiset<size_t, MyHash> map;
    size_t collision_val = 2 + map.bucket_count(); // try to get a bucket collision
    map.rehash(3);
    map.insert(1);
    map.insert(collision_val);
    map.insert(2);
    LIBCPP_ASSERT(map.bucket(2) == map.bucket(collision_val));

    auto erase = map.equal_range(2);
    map.erase(erase.first, erase.second);
    for (const auto& v : map)
      assert(v == 1 || v == collision_val);
  }
  { // Make sure that we're properly updating the bucket list when we're erasing to the end
    std::unordered_multiset<int> m;
    m.insert(1);
    m.insert(2);

    {
      auto pair = m.equal_range(1);
      assert(pair.first != pair.second);
      m.erase(pair.first, pair.second);
    }

    {
      auto pair = m.equal_range(2);
      assert(pair.first != pair.second);
      m.erase(pair.first, pair.second);
    }

    m.insert(3);
    assert(m.size() == 1);
    assert(*m.begin() == 3);
    assert(++m.begin() == m.end());
  }
#if TEST_STD_VER >= 11
  {
    typedef std::unordered_multiset<int, std::hash<int>, std::equal_to<int>, min_allocator<int>> C;
    typedef int P;
    P a[] = {P(1), P(2), P(3), P(4), P(1), P(2)};
    C c(a, a + sizeof(a) / sizeof(a[0]));
    C::const_iterator i = c.find(2);
    C::const_iterator j = std::next(i, 2);
    C::iterator k       = c.erase(i, i);
    assert(k == i);
    assert(c.size() == 6);
    assert(c.count(1) == 2);
    assert(c.count(2) == 2);
    assert(c.count(3) == 1);
    assert(c.count(4) == 1);

    k = c.erase(i, j);
    assert(c.size() == 4);
    assert(c.count(1) == 2);
    assert(c.count(3) == 1);
    assert(c.count(4) == 1);

    k = c.erase(c.cbegin(), c.cend());
    assert(c.size() == 0);
    assert(k == c.end());
  }
#endif

  return 0;
}
