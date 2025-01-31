//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template <class... Args>
//   pair<iterator, bool> emplace(Args&&... args);

#include <flat_set>
#include <cassert>
#include <deque>
#include <tuple>
#include <functional>
#include <vector>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "../../../Emplaceable.h"
#include "DefaultOnly.h"
#include "min_allocator.h"

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, std::less<Key>, KeyContainer>;
  using R   = std::pair<typename M::iterator, bool>;
  {
    // was empty
    M m;
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 1);
    assert(*r.first == 2);
  }
  {
    // key does not exist and inserted at the begin
    M m                              = {3, 5, 6, 7};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(r.second);
    assert(r.first == m.begin());
    assert(m.size() == 5);
    assert(*r.first == 2);
  }
  {
    // key does not exist and inserted in the middle
    M m                              = {0, 1, 3, 4};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(r.second);
    assert(r.first == m.begin() + 2);
    assert(m.size() == 5);
    assert(*r.first == 2);
  }
  {
    // key does not exist and inserted at the end
    M m                              = {0, 1};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(r.second);
    assert(r.first == m.begin() + 2);
    assert(m.size() == 3);
    assert(*r.first == 2);
  }
  {
    // key already exists and original at the begin
    M m                              = {2, 3, 5, 6};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(!r.second);
    assert(r.first == m.begin());
    assert(m.size() == 4);
    assert(*r.first == 2);
  }
  {
    // key already exists and original in the middle
    M m                              = {0, 2, 3, 4};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(!r.second);
    assert(r.first == m.begin() + 1);
    assert(m.size() == 4);
    assert(*r.first == 2);
  }
  {
    // key already exists and original at the end
    M m                              = {0, 1, 2};
    std::same_as<R> decltype(auto) r = m.emplace(typename M::value_type(2));
    assert(!r.second);
    assert(r.first == m.begin() + 2);
    assert(m.size() == 3);
    assert(*r.first == 2);
  }
}

template <class KeyContainer>
void test_emplaceable() {
  using M = std::flat_set<Emplaceable, std::less<Emplaceable>, KeyContainer>;
  using R = std::pair<typename M::iterator, bool>;

  M m;
  ASSERT_SAME_TYPE(decltype(m.emplace()), R);
  R r = m.emplace(2, 0.0);
  assert(r.second);
  assert(r.first == m.begin());
  assert(m.size() == 1);
  assert(*m.begin() == Emplaceable(2, 0.0));
  r = m.emplace(1, 3.5);
  assert(r.second);
  assert(r.first == m.begin());
  assert(m.size() == 2);
  assert(*m.begin() == Emplaceable(1, 3.5));
  r = m.emplace(1, 3.5);
  assert(!r.second);
  assert(r.first == m.begin());
  assert(m.size() == 2);
  assert(*m.begin() == Emplaceable(1, 3.5));
}

int main(int, char**) {
  test<std::vector<int>>();
  test<std::deque<int>>();
  test<MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>>();

  test_emplaceable<std::vector<Emplaceable>>();
  test_emplaceable<std::deque<Emplaceable>>();
  test_emplaceable<MinSequenceContainer<Emplaceable>>();
  test_emplaceable<std::vector<Emplaceable, min_allocator<Emplaceable>>>();

  {
    auto emplace_func = [](auto& m, auto key_arg) { m.emplace(key_arg); };
    test_emplace_exception_guarantee(emplace_func);
  }

  return 0;
}
