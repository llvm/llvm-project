//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// class flat_set

// pair<iterator, bool> insert( value_type&& v);

#include <flat_set>
#include <cassert>
#include <deque>

#include "MinSequenceContainer.h"
#include "MoveOnly.h"
#include "min_allocator.h"
#include "test_macros.h"
#include "../helpers.h"

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, TransparentComparator, KeyContainer>;
  using R   = std::pair<typename M::iterator, bool>;
  using V   = typename M::value_type;

  M m;
  std::same_as<R> decltype(auto) r = m.insert(V(2));
  assert(r.second);
  assert(r.first == m.begin());
  assert(m.size() == 1);
  assert(*r.first == V(2));

  r = m.insert(V(1));
  assert(r.second);
  assert(r.first == m.begin());
  assert(m.size() == 2);
  assert(*r.first == V(1));

  r = m.insert(V(3));
  assert(r.second);
  assert(r.first == std::ranges::prev(m.end()));
  assert(m.size() == 3);
  assert(*r.first == V(3));

  r = m.insert(V(3));
  assert(!r.second);
  assert(r.first == std::ranges::prev(m.end()));
  assert(m.size() == 3);
  assert(*r.first == V(3));
}

int main(int, char**) {
  test<std::vector<int>>();
  test<std::vector<MoveOnly>>();
  test<std::deque<int>>();
  test<std::deque<MoveOnly>>();
  test<MinSequenceContainer<int>>();
  test<MinSequenceContainer<MoveOnly>>();
  test<std::vector<int, min_allocator<int>>>();
  test<std::vector<MoveOnly, min_allocator<MoveOnly>>>();
  {
    auto insert_func = [](auto& m, auto key_arg) {
      using FlatSet    = std::decay_t<decltype(m)>;
      using value_type = typename FlatSet::value_type;
      value_type p(key_arg);
      m.insert(std::move(p));
    };
    test_emplace_exception_guarantee(insert_func);
  }

  return 0;
}
