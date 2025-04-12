//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// class flat_map

// pair<iterator, bool> insert( value_type&& v);

#include <flat_map>
#include <cassert>
#include <deque>

#include "MinSequenceContainer.h"
#include "MoveOnly.h"
#include "min_allocator.h"
#include "test_macros.h"
#include "../helpers.h"

template <class Container, class Pair>
void do_insert_rv_test() {
  using M = Container;
  using P = Pair;
  using R = std::pair<typename M::iterator, bool>;
  M m;
  std::same_as<R> decltype(auto) r = m.insert(P(2, 2));
  assert(r.second);
  assert(r.first == m.begin());
  assert(m.size() == 1);
  assert(r.first->first == 2);
  assert(r.first->second == 2);

  r = m.insert(P(1, 1));
  assert(r.second);
  assert(r.first == m.begin());
  assert(m.size() == 2);
  assert(r.first->first == 1);
  assert(r.first->second == 1);

  r = m.insert(P(3, 3));
  assert(r.second);
  assert(r.first == std::ranges::prev(m.end()));
  assert(m.size() == 3);
  assert(r.first->first == 3);
  assert(r.first->second == 3);

  r = m.insert(P(3, 3));
  assert(!r.second);
  assert(r.first == std::ranges::prev(m.end()));
  assert(m.size() == 3);
  assert(r.first->first == 3);
  assert(r.first->second == 3);
}

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, TransparentComparator, KeyContainer, ValueContainer>;

  using P  = std::pair<Key, Value>;
  using CP = std::pair<const Key, Value>;

  do_insert_rv_test<M, P>();
  do_insert_rv_test<M, CP>();
}

int main(int, char**) {
  test<std::vector<int>, std::vector<MoveOnly>>();
  test<std::deque<int>, std::vector<MoveOnly>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<MoveOnly>>();
  test<std::vector<int, min_allocator<int>>, std::vector<MoveOnly, min_allocator<MoveOnly>>>();

  {
    using M = std::flat_map<int, MoveOnly>;
    using R = std::pair<M::iterator, bool>;
    M m;
    R r = m.insert({2, MoveOnly(2)});
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 1);
    assert(r.first->first == 2);
    assert(r.first->second == 2);

    r = m.insert({1, MoveOnly(1)});
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 2);
    assert(r.first->first == 1);
    assert(r.first->second == 1);

    r = m.insert({3, MoveOnly(3)});
    assert(r.second);
    assert(r.first == std::ranges::prev(m.end()));
    assert(m.size() == 3);
    assert(r.first->first == 3);
    assert(r.first->second == 3);

    r = m.insert({3, MoveOnly(3)});
    assert(!r.second);
    assert(r.first == std::ranges::prev(m.end()));
    assert(m.size() == 3);
    assert(r.first->first == 3);
    assert(r.first->second == 3);
  }
  {
    auto insert_func = [](auto& m, auto key_arg, auto value_arg) {
      using FlatMap    = std::decay_t<decltype(m)>;
      using value_type = typename FlatMap::value_type;
      value_type p(std::piecewise_construct, std::tuple(key_arg), std::tuple(value_arg));
      m.insert(std::move(p));
    };
    test_emplace_exception_guarantee(insert_func);
  }

  return 0;
}
