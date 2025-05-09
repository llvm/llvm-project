//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// iterator insert(const_iterator position, const value_type& v);

#include <flat_set>
#include <cassert>
#include <functional>
#include <deque>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "../helpers.h"
#include "min_allocator.h"

template <class KeyContainer>
void test_one() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_multiset<Key, std::less<Key>, KeyContainer>;
  using R   = typename M::iterator;
  using VT  = typename M::value_type;

  M m;
  const VT v1(2);
  std::same_as<R> decltype(auto) r = m.insert(m.end(), v1);
  assert(r == m.begin());
  assert(m.size() == 1);
  assert(*r == 2);

  const VT v2(1);
  r = m.insert(m.end(), v2);
  assert(r == m.begin());
  assert(m.size() == 2);
  assert(*r == 1);

  const VT v3(3);
  r = m.insert(m.end(), v3);
  assert(r == std::ranges::prev(m.end()));
  assert(m.size() == 3);
  assert(*r == 3);

  const VT v4(3);
  r = m.insert(m.end(), v4);
  assert(r == std::ranges::prev(m.end()));
  assert(m.size() == 4);
  assert(*r == 3);

  const VT v5(1);
  r = m.insert(m.begin() + 2, v5);
  assert(r == m.begin() + 1);
  assert(m.size() == 5);
  assert(*r == 1);
}

void test() {
  test_one<std::vector<int>>();
  test_one<std::deque<int>>();
  test_one<MinSequenceContainer<int>>();
  test_one<std::vector<int, min_allocator<int>>>();
}

void test_exception() {
  auto insert_func = [](auto& m, auto key_arg) {
    using FlatSet    = std::decay_t<decltype(m)>;
    using value_type = typename FlatSet::value_type;
    const value_type p(key_arg);
    m.insert(m.begin(), p);
  };
  test_emplace_exception_guarantee(insert_func);
}

int main(int, char**) {
  test();
  test_exception();

  return 0;
}
