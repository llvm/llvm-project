//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>
//     iterator insert(const_iterator position, value_type&&);

#include <flat_set>
#include <cassert>
#include <deque>

#include "MinSequenceContainer.h"
#include "MoveOnly.h"
#include "min_allocator.h"
#include "../helpers.h"
#include "test_macros.h"

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, std::less<Key>, KeyContainer>;
  using V   = Key;
  using R   = typename M::iterator;
  M m;
  std::same_as<R> decltype(auto) r = m.insert(m.end(), V(2));
  assert(r == m.begin());
  assert(m.size() == 1);
  assert(*r == V(2));

  r = m.insert(m.end(), V(1));
  assert(r == m.begin());
  assert(m.size() == 2);
  assert(*r == V(1));

  r = m.insert(m.end(), V(3));
  assert(r == std::ranges::prev(m.end()));
  assert(m.size() == 3);
  assert(*r == V(3));

  r = m.insert(m.end(), V(3));
  assert(r == std::ranges::prev(m.end()));
  assert(m.size() == 3);
  assert(*r == V(3));
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
      m.insert(m.begin(), std::move(p));
    };
    test_emplace_exception_guarantee(insert_func);
  }

  return 0;
}
