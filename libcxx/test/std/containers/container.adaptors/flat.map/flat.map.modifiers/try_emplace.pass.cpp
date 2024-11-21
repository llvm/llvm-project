//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template<class... Args>
//   pair<iterator, bool> try_emplace(const key_type& k, Args&&... args);
// template<class... Args>
//   pair<iterator, bool> try_emplace(key_type&& k, Args&&... args);
// template<class... Args>
//   iterator try_emplace(const_iterator hint, const key_type& k, Args&&... args);
// template<class... Args>
//   iterator try_emplace(const_iterator hint, key_type&& k, Args&&... args);

#include <flat_map>
#include <cassert>
#include <functional>
#include <deque>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "../helpers.h"
#include "min_allocator.h"
#include "../../../Emplaceable.h"

// Constraints: is_constructible_v<mapped_type, Args...> is true.
template <class M, class... Args>
concept CanTryEmplace = requires(M m, Args&&... args) { m.try_emplace(std::forward<Args>(args)...); };

using Map  = std::flat_map<Emplaceable, Emplaceable>;
using Iter = typename Map::const_iterator;
static_assert(!CanTryEmplace<Map>);

static_assert(CanTryEmplace<Map, const Emplaceable&>);
static_assert(CanTryEmplace<Map, const Emplaceable&, Emplaceable>);
static_assert(CanTryEmplace<Map, const Emplaceable&, int, double>);
static_assert(!CanTryEmplace<Map, const Emplaceable&, const Emplaceable&>);
static_assert(!CanTryEmplace<Map, const Emplaceable&, int>);

static_assert(CanTryEmplace<Map, Emplaceable>);
static_assert(CanTryEmplace<Map, Emplaceable, Emplaceable>);
static_assert(CanTryEmplace<Map, Emplaceable, int, double>);
static_assert(!CanTryEmplace<Map, Emplaceable, const Emplaceable&>);
static_assert(!CanTryEmplace<Map, Emplaceable, int>);

static_assert(CanTryEmplace<Map, Iter, const Emplaceable&>);
static_assert(CanTryEmplace<Map, Iter, const Emplaceable&, Emplaceable>);
static_assert(CanTryEmplace<Map, Iter, const Emplaceable&, int, double>);
static_assert(!CanTryEmplace<Map, Iter, const Emplaceable&, const Emplaceable&>);
static_assert(!CanTryEmplace<Map, Iter, const Emplaceable&, int>);

static_assert(CanTryEmplace<Map, Iter, Emplaceable>);
static_assert(CanTryEmplace<Map, Iter, Emplaceable, Emplaceable>);
static_assert(CanTryEmplace<Map, Iter, Emplaceable, int, double>);
static_assert(!CanTryEmplace<Map, Iter, Emplaceable, const Emplaceable&>);
static_assert(!CanTryEmplace<Map, Iter, Emplaceable, int>);

template <class KeyContainer, class ValueContainer>
void test_ck() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;

  { // pair<iterator, bool> try_emplace(const key_type& k, Args&&... args);
    using R = std::pair<typename M::iterator, bool>;
    M m;
    for (int i = 0; i < 20; i += 2)
      m.emplace(i, Moveable(i, (double)i));

    assert(m.size() == 10);

    Moveable mv1(3, 3.0);
    for (int i = 0; i < 20; i += 2) {
      std::same_as<R> decltype(auto) r = m.try_emplace(i, std::move(mv1));
      assert(m.size() == 10);
      assert(!r.second);           // was not inserted
      assert(!mv1.moved());        // was not moved from
      assert(r.first->first == i); // key
    }

    std::same_as<R> decltype(auto) r2 = m.try_emplace(-1, std::move(mv1));
    assert(m.size() == 11);
    assert(r2.second);                   // was inserted
    assert(mv1.moved());                 // was moved from
    assert(r2.first->first == -1);       // key
    assert(r2.first->second.get() == 3); // value

    Moveable mv2(5, 3.0);
    std::same_as<R> decltype(auto) r3 = m.try_emplace(5, std::move(mv2));
    assert(m.size() == 12);
    assert(r3.second);                   // was inserted
    assert(mv2.moved());                 // was moved from
    assert(r3.first->first == 5);        // key
    assert(r3.first->second.get() == 5); // value

    Moveable mv3(-1, 3.0);
    std::same_as<R> decltype(auto) r4 = m.try_emplace(117, std::move(mv2));
    assert(m.size() == 13);
    assert(r4.second);                    // was inserted
    assert(mv2.moved());                  // was moved from
    assert(r4.first->first == 117);       // key
    assert(r4.first->second.get() == -1); // value
  }

  { // iterator try_emplace(const_iterator hint, const key_type& k, Args&&... args);
    using R = typename M::iterator;
    M m;
    for (int i = 0; i < 20; i += 2)
      m.try_emplace(i, Moveable(i, (double)i));
    assert(m.size() == 10);
    typename M::const_iterator it = m.find(2);

    Moveable mv1(3, 3.0);
    for (int i = 0; i < 20; i += 2) {
      std::same_as<R> decltype(auto) r1 = m.try_emplace(it, i, std::move(mv1));
      assert(m.size() == 10);
      assert(!mv1.moved());          // was not moved from
      assert(r1->first == i);        // key
      assert(r1->second.get() == i); // value
    }

    std::same_as<R> decltype(auto) r2 = m.try_emplace(it, 3, std::move(mv1));
    assert(m.size() == 11);
    assert(mv1.moved());           // was moved from
    assert(r2->first == 3);        // key
    assert(r2->second.get() == 3); // value
  }
}

template <class KeyContainer, class ValueContainer>
void test_rk() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;

  { // pair<iterator, bool> try_emplace(key_type&& k, Args&&... args);
    using R = std::pair<typename M::iterator, bool>;
    M m;
    for (int i = 0; i < 20; i += 2) {
      m.emplace(Moveable(i, (double)i), Moveable(i + 1, (double)i + 1));
    }
    assert(m.size() == 10);

    Moveable mvkey1(2, 2.0);
    Moveable mv1(4, 4.0);
    std::same_as<R> decltype(auto) r1 = m.try_emplace(std::move(mvkey1), std::move(mv1));
    assert(m.size() == 10);
    assert(!r1.second);                // was not inserted
    assert(!mv1.moved());              // was not moved from
    assert(!mvkey1.moved());           // was not moved from
    assert(r1.first->first == mvkey1); // key

    Moveable mvkey2(3, 3.0);
    std::same_as<R> decltype(auto) r2 = m.try_emplace(std::move(mvkey2), std::move(mv1));
    assert(m.size() == 11);
    assert(r2.second);                   // was inserted
    assert(mv1.moved());                 // was moved from
    assert(mvkey2.moved());              // was moved from
    assert(r2.first->first.get() == 3);  // key
    assert(r2.first->second.get() == 4); // value
  }

  { // iterator try_emplace(const_iterator hint, key_type&& k, Args&&... args);
    using R = typename M::iterator;
    M m;
    for (int i = 0; i < 20; i += 2)
      m.emplace(Moveable(i, (double)i), Moveable(i + 1, (double)i + 1));
    assert(m.size() == 10);
    typename M::const_iterator it = std::next(m.cbegin());

    Moveable mvkey1(2, 2.0);
    Moveable mv1(4, 4.0);
    std::same_as<R> decltype(auto) r1 = m.try_emplace(it, std::move(mvkey1), std::move(mv1));
    assert(m.size() == 10);
    assert(!mv1.moved());        // was not moved from
    assert(!mvkey1.moved());     // was not moved from
    assert(r1->first == mvkey1); // key

    Moveable mvkey2(3, 3.0);
    std::same_as<R> decltype(auto) r2 = m.try_emplace(it, std::move(mvkey2), std::move(mv1));
    assert(m.size() == 11);
    assert(mv1.moved());           // was moved from
    assert(mvkey2.moved());        // was moved from
    assert(r2->first.get() == 3);  // key
    assert(r2->second.get() == 4); // value
  }
}

int main(int, char**) {
  test_ck<std::vector<int>, std::vector<Moveable>>();
  test_ck<std::deque<int>, std::vector<Moveable>>();
  test_ck<MinSequenceContainer<int>, MinSequenceContainer<Moveable>>();
  test_ck<std::vector<int, min_allocator<int>>, std::vector<Moveable, min_allocator<Moveable>>>();

  test_rk<std::vector<Moveable>, std::vector<Moveable>>();
  test_rk<std::deque<Moveable>, std::vector<Moveable>>();
  test_rk<MinSequenceContainer<Moveable>, MinSequenceContainer<Moveable>>();
  test_rk<std::vector<Moveable, min_allocator<Moveable>>, std::vector<Moveable, min_allocator<Moveable>>>();

  {
    auto try_emplace_ck = [](auto& m, auto key_arg, auto value_arg) {
      using M   = std::decay_t<decltype(m)>;
      using Key = typename M::key_type;
      const Key key{key_arg};
      m.try_emplace(key, value_arg);
    };
    test_emplace_exception_guarantee(try_emplace_ck);
  }

  {
    auto try_emplace_rk = [](auto& m, auto key_arg, auto value_arg) {
      using M   = std::decay_t<decltype(m)>;
      using Key = typename M::key_type;
      m.try_emplace(Key{key_arg}, value_arg);
    };
    test_emplace_exception_guarantee(try_emplace_rk);
  }

  {
    auto try_emplace_iter_ck = [](auto& m, auto key_arg, auto value_arg) {
      using M   = std::decay_t<decltype(m)>;
      using Key = typename M::key_type;
      const Key key{key_arg};
      m.try_emplace(m.begin(), key, value_arg);
    };
    test_emplace_exception_guarantee(try_emplace_iter_ck);
  }

  {
    auto try_emplace_iter_rk = [](auto& m, auto key_arg, auto value_arg) {
      using M   = std::decay_t<decltype(m)>;
      using Key = typename M::key_type;
      m.try_emplace(m.begin(), Key{key_arg}, value_arg);
    };
    test_emplace_exception_guarantee(try_emplace_iter_rk);
  }

  return 0;
}
