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
//   iterator emplace_hint(const_iterator position, Args&&... args);

#include <flat_set>
#include <cassert>
#include <deque>
#include <functional>
#include <vector>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "../../../Emplaceable.h"
#include "DefaultOnly.h"
#include "min_allocator.h"
#include "../helpers.h"

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, std::less<Key>, KeyContainer>;
  using R   = M::iterator;
  {
    // was empty
    M m;
    std::same_as<R> decltype(auto) r = m.emplace_hint(m.end(), typename M::value_type(2));
    assert(r == m.begin());
    assert(m.size() == 1);
    assert(*r == 2);
  }
  {
    // hints correct at the begin
    M m                              = {3, 4};
    auto hint                        = m.begin();
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin());
    assert(m.size() == 3);
    assert(*r == 2);
  }
  {
    // hints correct in the middle
    M m                              = {0, 1, 3, 4};
    auto hint                        = m.begin() + 2;
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 5);
    assert(*r == 2);
  }
  {
    // hints correct at the end
    M m                              = {0, 1};
    auto hint                        = m.end();
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 3);
    assert(*r == 2);
  }
  {
    // hints correct but key already exists
    M m                              = {0, 1, 2, 3, 4};
    auto hint                        = m.begin() + 2;
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 5);
    assert(*r == 2);
  }
  {
    // hints incorrectly at the begin
    M m                              = {1, 4};
    auto hint                        = m.begin();
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 1);
    assert(m.size() == 3);
    assert(*r == 2);
  }
  {
    // hints incorrectly in the middle
    M m                              = {0, 1, 3, 4};
    auto hint                        = m.begin() + 1;
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 5);
    assert(*r == 2);
  }
  {
    // hints incorrectly at the end
    M m                              = {0, 3};
    auto hint                        = m.end();
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 1);
    assert(m.size() == 3);
    assert(*r == 2);
  }
  {
    // hints incorrect and key already exists
    M m                              = {0, 1, 2, 3, 4};
    auto hint                        = m.begin();
    std::same_as<R> decltype(auto) r = m.emplace_hint(hint, typename M::value_type(2));
    assert(r == m.begin() + 2);
    assert(m.size() == 5);
    assert(*r == 2);
  }
}

template <class KeyContainer>
void test_emplaceable() {
  using M = std::flat_set<Emplaceable, std::less<Emplaceable>, KeyContainer>;
  using R = M::iterator;

  M m;
  ASSERT_SAME_TYPE(decltype(m.emplace_hint(m.cbegin())), R);
  R r = m.emplace_hint(m.end(), 2, 0.0);
  assert(r == m.begin());
  assert(m.size() == 1);
  assert(*m.begin() == Emplaceable(2, 0.0));
  r = m.emplace_hint(m.end(), 1, 3.5);
  assert(r == m.begin());
  assert(m.size() == 2);
  assert(*m.begin() == Emplaceable(1, 3.5));
  r = m.emplace_hint(m.end(), 1, 3.5);
  assert(r == m.begin());
  assert(m.size() == 2);
  assert(*m.begin() == Emplaceable(1, 3.5));
}

int main(int, char**) {
  test<std::vector<int>>();
  test<std::deque<int>>();
  test<MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>>();

  test_emplaceable<std::vector<Emplaceable>>();
  test_emplaceable<std::vector<Emplaceable>>();
  test_emplaceable<MinSequenceContainer<Emplaceable>>();
  test_emplaceable<std::vector<Emplaceable, min_allocator<Emplaceable>>>();

  {
    auto emplace_func = [](auto& m, auto key_arg) { m.emplace_hint(m.begin(), key_arg); };
    test_emplace_exception_guarantee(emplace_func);
  }

  return 0;
}
