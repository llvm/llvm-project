//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_multiset

// template <class... Args>
//     iterator emplace(Args&&... args);

#include <cassert>
#include <unordered_set>

#include "../../Emplaceable.h"
#include "MoveOnly.h"
#include "min_allocator.h"

int main(int, char**) {
  {
    typedef std::unordered_multiset<Emplaceable> C;
    typedef C::iterator R;
    C c;
    R r = c.emplace();
    assert(c.size() == 1);
    assert(*r == Emplaceable());

    r = c.emplace(Emplaceable(5, 6));
    assert(c.size() == 2);
    assert(*r == Emplaceable(5, 6));

    r = c.emplace(5, 6);
    assert(c.size() == 3);
    assert(*r == Emplaceable(5, 6));
  }
  {
    typedef std::
        unordered_multiset<Emplaceable, std::hash<Emplaceable>, std::equal_to<Emplaceable>, min_allocator<Emplaceable>>
            C;
    typedef C::iterator R;
    C c;
    R r = c.emplace();
    assert(c.size() == 1);
    assert(*r == Emplaceable());

    r = c.emplace(Emplaceable(5, 6));
    assert(c.size() == 2);
    assert(*r == Emplaceable(5, 6));

    r = c.emplace(5, 6);
    assert(c.size() == 3);
    assert(*r == Emplaceable(5, 6));
  }
  { // We're unwrapping pairs for `unordered_{,multi}map`. Make sure we're not trying to do that for unordered_multiset.
    struct PairHasher {
      size_t operator()(const std::pair<MoveOnly, MoveOnly>& val) const { return std::hash<MoveOnly>()(val.first); }
    };
    using Set = std::unordered_multiset<std::pair<MoveOnly, MoveOnly>, PairHasher>;
    Set set;
    auto iter = set.emplace(std::pair<MoveOnly, MoveOnly>(2, 4));
    assert(set.begin() == iter);
  }

  return 0;
}
