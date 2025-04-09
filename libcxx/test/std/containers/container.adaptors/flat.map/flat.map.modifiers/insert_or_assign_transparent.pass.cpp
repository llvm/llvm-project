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

// template<class K, class M>
//   pair<iterator, bool> insert_or_assign(K&& k, M&& obj);
// template<class K, class M>
//   iterator insert_or_assign(const_iterator hint, K&& k, M&& obj);

// Constraints:
// The qualified-id Compare::is_transparent is valid and denotes a type.
// is_constructible_v<key_type, K> is true.
// is_assignable_v<mapped_type&, M> is true.
// is_constructible_v<mapped_type, M> is true.

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

static_assert(CanInsertOrAssign<std::flat_map<int, ConstructAndAssignFrom<V>, TransparentComparator>,
                                ConvertibleTransparent<int>,
                                V>);
static_assert(!CanInsertOrAssign<std::flat_map<int, ConstructAndAssignFrom<V>, TransparentComparator>,
                                 NonConvertibleTransparent<int>,
                                 V>);
static_assert(!CanInsertOrAssign<std::flat_map<int, ConstructAndAssignFrom<V>, NonTransparentComparator>,
                                 NonConvertibleTransparent<int>,
                                 V>);
static_assert(!CanInsertOrAssign<std::flat_map<int, ConstructAndAssignFrom<V>, TransparentComparator>,
                                 ConvertibleTransparent<int>,
                                 int>);
static_assert(
    !CanInsertOrAssign<std::flat_map<int, ConstructFrom<V>, TransparentComparator>, ConvertibleTransparent<int>, V>);
static_assert(
    !CanInsertOrAssign<std::flat_map<int, AssignFrom<V>, TransparentComparator>, ConvertibleTransparent<int>, V>);

static_assert(CanInsertOrAssignIter<std::flat_map<int, ConstructAndAssignFrom<V>, TransparentComparator>,
                                    ConvertibleTransparent<int>,
                                    V>);
static_assert(!CanInsertOrAssignIter<std::flat_map<int, ConstructAndAssignFrom<V>, TransparentComparator>,
                                     NonConvertibleTransparent<int>,
                                     V>);
static_assert(!CanInsertOrAssignIter<std::flat_map<int, ConstructAndAssignFrom<V>, NonTransparentComparator>,
                                     NonConvertibleTransparent<int>,
                                     V>);
static_assert(!CanInsertOrAssignIter<std::flat_map<int, ConstructAndAssignFrom<V>, TransparentComparator>,
                                     ConvertibleTransparent<int>,
                                     int>);
static_assert(!CanInsertOrAssignIter<std::flat_map<int, ConstructFrom<V>, TransparentComparator>,
                                     ConvertibleTransparent<int>,
                                     V>);
static_assert(
    !CanInsertOrAssignIter<std::flat_map<int, AssignFrom<V>, TransparentComparator>, ConvertibleTransparent<int>, V>);

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, TransparentComparator, KeyContainer, ValueContainer>;
  {
    // pair<iterator, bool> insert_or_assign(const key_type& k, M&& obj);
    using R = std::pair<typename M::iterator, bool>;
    M m;
    for (int i = 0; i < 20; i += 2)
      m.emplace(i, Moveable(i, (double)i));
    assert(m.size() == 10);

    for (int i = 0; i < 20; i += 2) {
      Moveable mv(i + 1, i + 1);
      std::same_as<R> decltype(auto) r1 = m.insert_or_assign(ConvertibleTransparent<int>{i}, std::move(mv));
      assert(m.size() == 10);
      assert(!r1.second);                      // was not inserted
      assert(mv.moved());                      // was moved from
      assert(r1.first->first == i);            // key
      assert(r1.first->second.get() == i + 1); // value
    }

    Moveable mv1(5, 5.0);
    std::same_as<R> decltype(auto) r2 = m.insert_or_assign(ConvertibleTransparent<int>{-1}, std::move(mv1));
    assert(m.size() == 11);
    assert(r2.second);                   // was inserted
    assert(mv1.moved());                 // was moved from
    assert(r2.first->first == -1);       // key
    assert(r2.first->second.get() == 5); // value

    Moveable mv2(9, 9.0);
    std::same_as<R> decltype(auto) r3 = m.insert_or_assign(ConvertibleTransparent<int>{3}, std::move(mv2));
    assert(m.size() == 12);
    assert(r3.second);                   // was inserted
    assert(mv2.moved());                 // was moved from
    assert(r3.first->first == 3);        // key
    assert(r3.first->second.get() == 9); // value

    Moveable mv3(-1, 5.0);
    std::same_as<R> decltype(auto) r4 = m.insert_or_assign(ConvertibleTransparent<int>{117}, std::move(mv3));
    assert(m.size() == 13);
    assert(r4.second);                    // was inserted
    assert(mv3.moved());                  // was moved from
    assert(r4.first->first == 117);       // key
    assert(r4.first->second.get() == -1); // value
  }
  {
    // iterator insert_or_assign(const_iterator hint, const key_type& k, M&& obj);
    using R = M::iterator;
    M m;
    for (int i = 0; i < 20; i += 2)
      m.emplace(i, Moveable(i, (double)i));
    assert(m.size() == 10);
    typename M::const_iterator it = m.find(2);

    Moveable mv1(3, 3.0);
    std::same_as<R> decltype(auto) r1 = m.insert_or_assign(it, ConvertibleTransparent<int>{2}, std::move(mv1));
    assert(m.size() == 10);
    assert(mv1.moved());           // was moved from
    assert(r1->first == 2);        // key
    assert(r1->second.get() == 3); // value

    Moveable mv2(5, 5.0);
    std::same_as<R> decltype(auto) r2 = m.insert_or_assign(it, ConvertibleTransparent<int>{3}, std::move(mv2));
    assert(m.size() == 11);
    assert(mv2.moved());           // was moved from
    assert(r2->first == 3);        // key
    assert(r2->second.get() == 5); // value

    // wrong hint: begin()
    Moveable mv3(7, 7.0);
    std::same_as<R> decltype(auto) r3 = m.insert_or_assign(m.begin(), ConvertibleTransparent<int>{4}, std::move(mv3));
    assert(m.size() == 11);
    assert(mv3.moved());           // was moved from
    assert(r3->first == 4);        // key
    assert(r3->second.get() == 7); // value

    Moveable mv4(9, 9.0);
    std::same_as<R> decltype(auto) r4 = m.insert_or_assign(m.begin(), ConvertibleTransparent<int>{5}, std::move(mv4));
    assert(m.size() == 12);
    assert(mv4.moved());           // was moved from
    assert(r4->first == 5);        // key
    assert(r4->second.get() == 9); // value

    // wrong hint: end()
    Moveable mv5(11, 11.0);
    std::same_as<R> decltype(auto) r5 = m.insert_or_assign(m.end(), ConvertibleTransparent<int>{6}, std::move(mv5));
    assert(m.size() == 12);
    assert(mv5.moved());            // was moved from
    assert(r5->first == 6);         // key
    assert(r5->second.get() == 11); // value

    Moveable mv6(13, 13.0);
    std::same_as<R> decltype(auto) r6 = m.insert_or_assign(m.end(), ConvertibleTransparent<int>{7}, std::move(mv6));
    assert(m.size() == 13);
    assert(mv6.moved());            // was moved from
    assert(r6->first == 7);         // key
    assert(r6->second.get() == 13); // value

    // wrong hint: third element
    Moveable mv7(15, 15.0);
    std::same_as<R> decltype(auto) r7 =
        m.insert_or_assign(std::next(m.begin(), 2), ConvertibleTransparent<int>{8}, std::move(mv7));
    assert(m.size() == 13);
    assert(mv7.moved());            // was moved from
    assert(r7->first == 8);         // key
    assert(r7->second.get() == 15); // value

    Moveable mv8(17, 17.0);
    std::same_as<R> decltype(auto) r8 =
        m.insert_or_assign(std::next(m.begin(), 2), ConvertibleTransparent<int>{9}, std::move(mv8));
    assert(m.size() == 14);
    assert(mv8.moved());            // was moved from
    assert(r8->first == 9);         // key
    assert(r8->second.get() == 17); // value
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
    auto p = m.insert_or_assign(ConvertibleTransparent<int>{3}, 5);
    assert(!p.second);
    assert(transparent_used);
  }
  {
    bool transparent_used = false;
    TransparentComparator c(transparent_used);
    std::flat_map<int, int, TransparentComparator> m(std::sorted_unique, {{1, 1}, {2, 2}, {3, 3}}, c);
    assert(!transparent_used);
    auto it = m.insert_or_assign(m.begin(), ConvertibleTransparent<int>{3}, 5);
    assert(it->second == 5);
    assert(transparent_used);
  }

  {
    auto insert_or_assign = [](auto& m, auto key_arg, auto value_arg) {
      using M   = std::decay_t<decltype(m)>;
      using Key = typename M::key_type;
      m.insert_or_assign(ConvertibleTransparent<Key>{key_arg}, value_arg);
    };
    test_emplace_exception_guarantee(insert_or_assign);
  }

  {
    auto insert_or_assign_iter = [](auto& m, auto key_arg, auto value_arg) {
      using M   = std::decay_t<decltype(m)>;
      using Key = typename M::key_type;
      m.insert_or_assign(m.begin(), ConvertibleTransparent<Key>{key_arg}, value_arg);
    };
    test_emplace_exception_guarantee(insert_or_assign_iter);
  }

  return 0;
}
