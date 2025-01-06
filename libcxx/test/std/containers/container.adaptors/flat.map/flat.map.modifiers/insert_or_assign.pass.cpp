//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

#include <flat_map>
#include <cassert>
#include <deque>

#include "MinSequenceContainer.h"
#include "MoveOnly.h"
#include "min_allocator.h"
#include "test_macros.h"
#include "../helpers.h"

// template<class M>
//   pair<iterator, bool> insert_or_assign(const key_type& k, M&& obj);
// template<class M>
//   pair<iterator, bool> insert_or_assign(key_type&& k, M&& obj);
// template<class M>
//   iterator insert_or_assign(const_iterator hint, const key_type& k, M&& obj);
// template<class M>
//   iterator insert_or_assign(const_iterator hint, key_type&& k, M&& obj);

// Constraints: is_assignable_v<mapped_type&, M> is true and is_constructible_v<mapped_type, M> is true.
template <class Map, class K, class M>
concept CanInsertOrAssign =
    requires(Map map, K&& k, M&& m) { map.insert_or_assign(std::forward<K>(k), std::forward<M>(m)); };

template <class Map, class K, class M>
concept CanInsertOrAssignIter = requires(Map map, typename Map::const_iterator iter, K&& k, M&& m) {
  map.insert_or_assign(iter, std::forward<K>(k), std::forward<M>(m));
};

template <class From>
struct ConstructAndAssignFrom {
  explicit ConstructAndAssignFrom(From);
  ConstructAndAssignFrom& operator=(From);
};

template <class From>
struct ConstructFrom {
  explicit ConstructFrom(From);
};

template <class From>
struct AssignFrom {
  AssignFrom& operator=(From);
};

struct V {};

static_assert(CanInsertOrAssign<std::flat_map<int, ConstructAndAssignFrom<V>>, const int&, V>);
static_assert(!CanInsertOrAssign<std::flat_map<int, ConstructAndAssignFrom<V>>, const int&, int>);
static_assert(!CanInsertOrAssign<std::flat_map<int, ConstructFrom<V>>, const int&, V>);
static_assert(!CanInsertOrAssign<std::flat_map<int, AssignFrom<V>>, const int&, V>);

static_assert(CanInsertOrAssign<std::flat_map<int, ConstructAndAssignFrom<V>>, int&&, V>);
static_assert(!CanInsertOrAssign<std::flat_map<int, ConstructAndAssignFrom<V>>, int&&, int>);
static_assert(!CanInsertOrAssign<std::flat_map<int, ConstructFrom<V>>, int&&, V>);
static_assert(!CanInsertOrAssign<std::flat_map<int, AssignFrom<V>>, int&&, V>);

static_assert(CanInsertOrAssignIter<std::flat_map<int, ConstructAndAssignFrom<V>>, const int&, V>);
static_assert(!CanInsertOrAssignIter<std::flat_map<int, ConstructAndAssignFrom<V>>, const int&, int>);
static_assert(!CanInsertOrAssignIter<std::flat_map<int, ConstructFrom<V>>, const int&, V>);
static_assert(!CanInsertOrAssignIter<std::flat_map<int, AssignFrom<V>>, const int&, V>);

static_assert(CanInsertOrAssignIter<std::flat_map<int, ConstructAndAssignFrom<V>>, int&&, V>);
static_assert(!CanInsertOrAssignIter<std::flat_map<int, ConstructAndAssignFrom<V>>, int&&, int>);
static_assert(!CanInsertOrAssignIter<std::flat_map<int, ConstructFrom<V>>, int&&, V>);
static_assert(!CanInsertOrAssignIter<std::flat_map<int, AssignFrom<V>>, int&&, V>);

template <class KeyContainer, class ValueContainer>
void test_cv_key() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, TransparentComparator, KeyContainer, ValueContainer>;
  { // pair<iterator, bool> insert_or_assign(const key_type& k, M&& obj);
    using R = std::pair<typename M::iterator, bool>;
    M m;
    for (int i = 0; i < 20; i += 2)
      m.emplace(i, Moveable(i, (double)i));
    assert(m.size() == 10);

    for (int i = 0; i < 20; i += 2) {
      Moveable mv(i + 1, i + 1);
      std::same_as<R> decltype(auto) r1 = m.insert_or_assign(i, std::move(mv));
      assert(m.size() == 10);
      assert(!r1.second);                      // was not inserted
      assert(mv.moved());                      // was moved from
      assert(r1.first->first == i);            // key
      assert(r1.first->second.get() == i + 1); // value
    }

    Moveable mv1(5, 5.0);
    std::same_as<R> decltype(auto) r2 = m.insert_or_assign(-1, std::move(mv1));
    assert(m.size() == 11);
    assert(r2.second);                   // was inserted
    assert(mv1.moved());                 // was moved from
    assert(r2.first->first == -1);       // key
    assert(r2.first->second.get() == 5); // value

    Moveable mv2(9, 9.0);
    std::same_as<R> decltype(auto) r3 = m.insert_or_assign(3, std::move(mv2));
    assert(m.size() == 12);
    assert(r3.second);                   // was inserted
    assert(mv2.moved());                 // was moved from
    assert(r3.first->first == 3);        // key
    assert(r3.first->second.get() == 9); // value

    Moveable mv3(-1, 5.0);
    std::same_as<R> decltype(auto) r4 = m.insert_or_assign(117, std::move(mv3));
    assert(m.size() == 13);
    assert(r4.second);                    // was inserted
    assert(mv3.moved());                  // was moved from
    assert(r4.first->first == 117);       // key
    assert(r4.first->second.get() == -1); // value
  }

  { // iterator insert_or_assign(const_iterator hint, const key_type& k, M&& obj);
    M m;
    using R = M::iterator;
    for (int i = 0; i < 20; i += 2)
      m.emplace(i, Moveable(i, (double)i));
    assert(m.size() == 10);
    typename M::const_iterator it = m.find(2);

    Moveable mv1(3, 3.0);
    std::same_as<R> decltype(auto) r1 = m.insert_or_assign(it, 2, std::move(mv1));
    assert(m.size() == 10);
    assert(mv1.moved());           // was moved from
    assert(r1->first == 2);        // key
    assert(r1->second.get() == 3); // value

    Moveable mv2(5, 5.0);
    std::same_as<R> decltype(auto) r2 = m.insert_or_assign(it, 3, std::move(mv2));
    assert(m.size() == 11);
    assert(mv2.moved());           // was moved from
    assert(r2->first == 3);        // key
    assert(r2->second.get() == 5); // value

    // wrong hint: begin()
    Moveable mv3(7, 7.0);
    std::same_as<R> decltype(auto) r3 = m.insert_or_assign(m.begin(), 4, std::move(mv3));
    assert(m.size() == 11);
    assert(mv3.moved());           // was moved from
    assert(r3->first == 4);        // key
    assert(r3->second.get() == 7); // value

    Moveable mv4(9, 9.0);
    std::same_as<R> decltype(auto) r4 = m.insert_or_assign(m.begin(), 5, std::move(mv4));
    assert(m.size() == 12);
    assert(mv4.moved());           // was moved from
    assert(r4->first == 5);        // key
    assert(r4->second.get() == 9); // value

    // wrong hint: end()
    Moveable mv5(11, 11.0);
    std::same_as<R> decltype(auto) r5 = m.insert_or_assign(m.end(), 6, std::move(mv5));
    assert(m.size() == 12);
    assert(mv5.moved());            // was moved from
    assert(r5->first == 6);         // key
    assert(r5->second.get() == 11); // value

    Moveable mv6(13, 13.0);
    std::same_as<R> decltype(auto) r6 = m.insert_or_assign(m.end(), 7, std::move(mv6));
    assert(m.size() == 13);
    assert(mv6.moved());            // was moved from
    assert(r6->first == 7);         // key
    assert(r6->second.get() == 13); // value

    // wrong hint: third element
    Moveable mv7(15, 15.0);
    std::same_as<R> decltype(auto) r7 = m.insert_or_assign(std::next(m.begin(), 2), 8, std::move(mv7));
    assert(m.size() == 13);
    assert(mv7.moved());            // was moved from
    assert(r7->first == 8);         // key
    assert(r7->second.get() == 15); // value

    Moveable mv8(17, 17.0);
    std::same_as<R> decltype(auto) r8 = m.insert_or_assign(std::next(m.begin(), 2), 9, std::move(mv8));
    assert(m.size() == 14);
    assert(mv8.moved());            // was moved from
    assert(r8->first == 9);         // key
    assert(r8->second.get() == 17); // value
  }
}

template <class KeyContainer, class ValueContainer>
void test_rv_key() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, TransparentComparator, KeyContainer, ValueContainer>;

  { // pair<iterator, bool> insert_or_assign(key_type&& k, M&& obj);
    using R = std::pair<typename M::iterator, bool>;
    M m;
    for (int i = 0; i < 20; i += 2)
      m.emplace(Moveable(i, (double)i), Moveable(i + 1, (double)i + 1));
    assert(m.size() == 10);

    Moveable mvkey1(2, 2.0);
    Moveable mv1(4, 4.0);
    std::same_as<R> decltype(auto) r1 = m.insert_or_assign(std::move(mvkey1), std::move(mv1));
    assert(m.size() == 10);
    assert(!r1.second);                  // was not inserted
    assert(!mvkey1.moved());             // was not moved from
    assert(mv1.moved());                 // was moved from
    assert(r1.first->first == mvkey1);   // key
    assert(r1.first->second.get() == 4); // value

    Moveable mvkey2(3, 3.0);
    Moveable mv2(5, 5.0);
    std::same_as<R> decltype(auto) r2 = m.try_emplace(std::move(mvkey2), std::move(mv2));
    assert(m.size() == 11);
    assert(r2.second);                   // was inserted
    assert(mv2.moved());                 // was moved from
    assert(mvkey2.moved());              // was moved from
    assert(r2.first->first.get() == 3);  // key
    assert(r2.first->second.get() == 5); // value
  }
  { // iterator insert_or_assign(const_iterator hint, key_type&& k, M&& obj);
    using R = M::iterator;
    M m;
    for (int i = 0; i < 20; i += 2)
      m.emplace(Moveable(i, (double)i), Moveable(i + 1, (double)i + 1));
    assert(m.size() == 10);
    typename M::const_iterator it = std::next(m.cbegin());

    Moveable mvkey1(2, 2.0);
    Moveable mv1(4, 4.0);
    std::same_as<R> decltype(auto) r1 = m.insert_or_assign(it, std::move(mvkey1), std::move(mv1));
    assert(m.size() == 10);
    assert(mv1.moved());           // was moved from
    assert(!mvkey1.moved());       // was not moved from
    assert(r1->first == mvkey1);   // key
    assert(r1->second.get() == 4); // value

    Moveable mvkey2(3, 3.0);
    Moveable mv2(5, 5.0);
    std::same_as<R> decltype(auto) r2 = m.insert_or_assign(it, std::move(mvkey2), std::move(mv2));
    assert(m.size() == 11);
    assert(mv2.moved());           // was moved from
    assert(mvkey2.moved());        // was moved from
    assert(r2->first.get() == 3);  // key
    assert(r2->second.get() == 5); // value

    // wrong hint: begin()
    Moveable mvkey3(6, 6.0);
    Moveable mv3(8, 8.0);
    std::same_as<R> decltype(auto) r3 = m.insert_or_assign(m.begin(), std::move(mvkey3), std::move(mv3));
    assert(m.size() == 11);
    assert(mv3.moved());           // was moved from
    assert(!mvkey3.moved());       // was not moved from
    assert(r3->first == mvkey3);   // key
    assert(r3->second.get() == 8); // value

    Moveable mvkey4(7, 7.0);
    Moveable mv4(9, 9.0);
    std::same_as<R> decltype(auto) r4 = m.insert_or_assign(m.begin(), std::move(mvkey4), std::move(mv4));
    assert(m.size() == 12);
    assert(mv4.moved());           // was moved from
    assert(mvkey4.moved());        // was moved from
    assert(r4->first.get() == 7);  // key
    assert(r4->second.get() == 9); // value

    // wrong hint: end()
    Moveable mvkey5(8, 8.0);
    Moveable mv5(10, 10.0);
    std::same_as<R> decltype(auto) r5 = m.insert_or_assign(m.end(), std::move(mvkey5), std::move(mv5));
    assert(m.size() == 12);
    assert(mv5.moved());            // was moved from
    assert(!mvkey5.moved());        // was not moved from
    assert(r5->first == mvkey5);    // key
    assert(r5->second.get() == 10); // value

    Moveable mvkey6(9, 9.0);
    Moveable mv6(11, 11.0);
    std::same_as<R> decltype(auto) r6 = m.insert_or_assign(m.end(), std::move(mvkey6), std::move(mv6));
    assert(m.size() == 13);
    assert(mv6.moved());            // was moved from
    assert(mvkey6.moved());         // was moved from
    assert(r6->first.get() == 9);   // key
    assert(r6->second.get() == 11); // value

    // wrong hint: third element
    Moveable mvkey7(10, 10.0);
    Moveable mv7(12, 12.0);
    std::same_as<R> decltype(auto) r7 = m.insert_or_assign(std::next(m.begin(), 2), std::move(mvkey7), std::move(mv7));
    assert(m.size() == 13);
    assert(mv7.moved());            // was moved from
    assert(!mvkey7.moved());        // was not moved from
    assert(r7->first == mvkey7);    // key
    assert(r7->second.get() == 12); // value

    Moveable mvkey8(11, 11.0);
    Moveable mv8(13, 13.0);
    std::same_as<R> decltype(auto) r8 = m.insert_or_assign(std::next(m.begin(), 2), std::move(mvkey8), std::move(mv8));
    assert(m.size() == 14);
    assert(mv8.moved());            // was moved from
    assert(mvkey8.moved());         // was moved from
    assert(r8->first.get() == 11);  // key
    assert(r8->second.get() == 13); // value
  }
}

int main(int, char**) {
  test_cv_key<std::vector<int>, std::vector<Moveable>>();
  test_cv_key<std::deque<int>, std::vector<Moveable>>();
  test_cv_key<MinSequenceContainer<int>, MinSequenceContainer<Moveable>>();
  test_cv_key<std::vector<int, min_allocator<int>>, std::vector<Moveable, min_allocator<Moveable>>>();

  test_rv_key<std::vector<Moveable>, std::vector<Moveable>>();
  test_rv_key<std::deque<Moveable>, std::vector<Moveable>>();
  test_rv_key<MinSequenceContainer<Moveable>, MinSequenceContainer<Moveable>>();
  test_rv_key<std::vector<Moveable, min_allocator<Moveable>>, std::vector<Moveable, min_allocator<Moveable>>>();

  return 0;
}
