//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// class flat_multimap

// iterator insert(const_iterator position, value_type&&);

#include <flat_map>
#include <cassert>
#include <deque>

#include "MinSequenceContainer.h"
#include "MoveOnly.h"
#include "min_allocator.h"
#include "../helpers.h"
#include "test_macros.h"

template <class Container, class Pair>
void do_insert_iter_rv_test() {
  using M = Container;
  using P = Pair;
  using R = typename M::iterator;
  M m;
  std::same_as<R> decltype(auto) r = m.insert(m.end(), P(2, 2));
  assert(r == m.begin());
  assert(m.size() == 1);
  assert(r->first == 2);
  assert(r->second == 2);

  r = m.insert(m.end(), P(1, 1));
  assert(r == m.begin());
  assert(m.size() == 2);
  assert(r->first == 1);
  assert(r->second == 1);

  r = m.insert(m.end(), P(3, 3));
  assert(r == std::ranges::prev(m.end()));
  assert(m.size() == 3);
  assert(r->first == 3);
  assert(r->second == 3);

  r = m.insert(m.end(), P(3, 4));
  assert(r == std::ranges::prev(m.end()));
  assert(m.size() == 4);
  assert(r->first == 3);
  assert(r->second == 4);

  r = m.insert(m.end(), P(2, 5));
  assert(r == m.begin() + 2);
  assert(m.size() == 5);
  assert(r->first == 2);
  assert(r->second == 5);

  r = m.insert(m.begin(), P(2, 6));
  assert(r == m.begin() + 1);
  assert(m.size() == 6);
  assert(r->first == 2);
  assert(r->second == 6);
}

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
  using P     = std::pair<Key, Value>;
  using CP    = std::pair<const Key, Value>;

  do_insert_iter_rv_test<M, P>();
  do_insert_iter_rv_test<M, CP>();
}

int main(int, char**) {
  test<std::vector<int>, std::vector<double>>();
  test<std::vector<int>, std::vector<MoveOnly>>();
  test<std::deque<int>, std::deque<double>>();
  test<std::deque<int>, std::deque<MoveOnly>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<MoveOnly>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();
  test<std::vector<int, min_allocator<int>>, std::vector<MoveOnly, min_allocator<MoveOnly>>>();

  {
    auto insert_func = [](auto& m, auto key_arg, auto value_arg) {
      using FlatMap    = std::decay_t<decltype(m)>;
      using value_type = typename FlatMap::value_type;
      value_type p(std::piecewise_construct, std::tuple(key_arg), std::tuple(value_arg));
      m.insert(m.begin(), std::move(p));
    };
    test_emplace_exception_guarantee(insert_func);
  }

  return 0;
}
