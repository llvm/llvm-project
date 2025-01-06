//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class K, class... Args>
//   pair<iterator, bool> try_emplace(K&& k, Args&&... args);
// template<class K, class... Args>
//   iterator try_emplace(const_iterator hint, K&& k, Args&&... args);

#include <flat_map>
#include <cassert>
#include <functional>
#include <deque>
#include <vector>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "../helpers.h"
#include "min_allocator.h"
#include "../../../Emplaceable.h"

// Constraints:
// The qualified-id Compare::is_transparent is valid and denotes a type.
// is_constructible_v<key_type, K> is true.
// is_constructible_v<mapped_type, Args...> is true.
// For the first overload, is_convertible_v<K&&, const_iterator> and is_convertible_v<K&&, iterator> are both false
template <class M, class... Args>
concept CanTryEmplace = requires(M m, Args&&... args) { m.try_emplace(std::forward<Args>(args)...); };

using TransparentMap    = std::flat_map<int, Emplaceable, TransparentComparator>;
using NonTransparentMap = std::flat_map<int, Emplaceable, NonTransparentComparator>;

using TransparentMapIter      = typename TransparentMap::iterator;
using TransparentMapConstIter = typename TransparentMap::const_iterator;

static_assert(!CanTryEmplace<TransparentMap>);
static_assert(!CanTryEmplace<NonTransparentMap>);

static_assert(CanTryEmplace<TransparentMap, ConvertibleTransparent<int>>);
static_assert(CanTryEmplace<TransparentMap, ConvertibleTransparent<int>, Emplaceable>);
static_assert(CanTryEmplace<TransparentMap, ConvertibleTransparent<int>, int, double>);
static_assert(!CanTryEmplace<TransparentMap, ConvertibleTransparent<int>, const Emplaceable&>);
static_assert(!CanTryEmplace<TransparentMap, ConvertibleTransparent<int>, int>);
static_assert(!CanTryEmplace<TransparentMap, NonConvertibleTransparent<int>, Emplaceable>);
static_assert(!CanTryEmplace<NonTransparentMap, NonConvertibleTransparent<int>, Emplaceable>);
static_assert(!CanTryEmplace<TransparentMap, ConvertibleTransparent<int>, int>);
static_assert(!CanTryEmplace<TransparentMap, TransparentMapIter, Emplaceable>);
static_assert(!CanTryEmplace<TransparentMap, TransparentMapConstIter, Emplaceable>);

static_assert(CanTryEmplace<TransparentMap, TransparentMapConstIter, ConvertibleTransparent<int>>);
static_assert(CanTryEmplace<TransparentMap, TransparentMapConstIter, ConvertibleTransparent<int>, Emplaceable>);
static_assert(CanTryEmplace<TransparentMap, TransparentMapConstIter, ConvertibleTransparent<int>, int, double>);
static_assert(!CanTryEmplace<TransparentMap, TransparentMapConstIter, ConvertibleTransparent<int>, const Emplaceable&>);
static_assert(!CanTryEmplace<TransparentMap, TransparentMapConstIter, ConvertibleTransparent<int>, int>);
static_assert(!CanTryEmplace<TransparentMap, TransparentMapConstIter, NonConvertibleTransparent<int>, Emplaceable>);
static_assert(!CanTryEmplace<NonTransparentMap, TransparentMapConstIter, NonConvertibleTransparent<int>, Emplaceable>);
static_assert(!CanTryEmplace<TransparentMap, TransparentMapConstIter, ConvertibleTransparent<int>, int>);

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, TransparentComparator, KeyContainer, ValueContainer>;

  { // pair<iterator, bool> try_emplace(K&& k, Args&&... args);
    using R = std::pair<typename M::iterator, bool>;
    M m;
    for (int i = 0; i < 20; i += 2)
      m.emplace(i, Moveable(i, (double)i));

    assert(m.size() == 10);

    Moveable mv1(3, 3.0);
    for (int i = 0; i < 20; i += 2) {
      std::same_as<R> decltype(auto) r = m.try_emplace(ConvertibleTransparent<int>{i}, std::move(mv1));
      assert(m.size() == 10);
      assert(!r.second);           // was not inserted
      assert(!mv1.moved());        // was not moved from
      assert(r.first->first == i); // key
    }

    std::same_as<R> decltype(auto) r2 = m.try_emplace(ConvertibleTransparent<int>{-1}, std::move(mv1));
    assert(m.size() == 11);
    assert(r2.second);                   // was inserted
    assert(mv1.moved());                 // was moved from
    assert(r2.first->first == -1);       // key
    assert(r2.first->second.get() == 3); // value

    Moveable mv2(5, 3.0);
    std::same_as<R> decltype(auto) r3 = m.try_emplace(ConvertibleTransparent<int>{5}, std::move(mv2));
    assert(m.size() == 12);
    assert(r3.second);                   // was inserted
    assert(mv2.moved());                 // was moved from
    assert(r3.first->first == 5);        // key
    assert(r3.first->second.get() == 5); // value

    Moveable mv3(-1, 3.0);
    std::same_as<R> decltype(auto) r4 = m.try_emplace(ConvertibleTransparent<int>{117}, std::move(mv2));
    assert(m.size() == 13);
    assert(r4.second);                    // was inserted
    assert(mv2.moved());                  // was moved from
    assert(r4.first->first == 117);       // key
    assert(r4.first->second.get() == -1); // value
  }

  { // iterator try_emplace(const_iterator hint, K&& k, Args&&... args);
    using R = typename M::iterator;
    M m;
    for (int i = 0; i < 20; i += 2)
      m.try_emplace(i, Moveable(i, (double)i));
    assert(m.size() == 10);
    typename M::const_iterator it = m.find(2);

    Moveable mv1(3, 3.0);
    for (int i = 0; i < 20; i += 2) {
      std::same_as<R> decltype(auto) r1 = m.try_emplace(it, ConvertibleTransparent<int>{i}, std::move(mv1));
      assert(m.size() == 10);
      assert(!mv1.moved());          // was not moved from
      assert(r1->first == i);        // key
      assert(r1->second.get() == i); // value
    }

    std::same_as<R> decltype(auto) r2 = m.try_emplace(it, ConvertibleTransparent<int>{3}, std::move(mv1));
    assert(m.size() == 11);
    assert(mv1.moved());           // was moved from
    assert(r2->first == 3);        // key
    assert(r2->second.get() == 3); // value
  }
}

int main(int, char**) {
  test<std::vector<int>, std::vector<Moveable>>();
  test<std::deque<int>, std::vector<Moveable>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<Moveable>>();
  test<std::vector<int, min_allocator<int>>, std::vector<Moveable, min_allocator<Moveable>>>();

  {
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    std::flat_map<int, int, TransparentComparator> m(std::sorted_unique, {{1, 1}, {2, 2}, {3, 3}}, c);
    assert(!transparent_used);
    auto p = m.try_emplace(ConvertibleTransparent<int>{3}, 3);
    assert(!p.second);
    assert(transparent_used);
  }
  {
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    std::flat_map<int, int, TransparentComparator> m(std::sorted_unique, {{1, 1}, {2, 2}, {3, 3}}, c);
    assert(!transparent_used);
    auto p = m.try_emplace(m.begin(), ConvertibleTransparent<int>{3}, 3);
    assert(p->second == 3);
    assert(transparent_used);
  }
  {
    auto try_emplace = [](auto& m, auto key_arg, auto value_arg) {
      using M   = std::decay_t<decltype(m)>;
      using Key = typename M::key_type;
      m.try_emplace(ConvertibleTransparent<Key>{key_arg}, value_arg);
    };
    test_emplace_exception_guarantee(try_emplace);
  }

  {
    auto try_emplace_iter = [](auto& m, auto key_arg, auto value_arg) {
      using M   = std::decay_t<decltype(m)>;
      using Key = typename M::key_type;
      m.try_emplace(m.begin(), ConvertibleTransparent<Key>{key_arg}, value_arg);
    };
    test_emplace_exception_guarantee(try_emplace_iter);
  }

  return 0;
}
