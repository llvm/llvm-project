//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// template <class... Args>
//   iterator emplace_hint(const_iterator position, Args&&... args);

#include <flat_map>
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

#if defined(_LIBCPP_VERSION)
// spec only specifies `emplace(Args&&...)` is_constructible_v<pair<key_type, mapped_type>, Args...> is true.
// nothing mentioned for emplace_hint
template <class M, class... Args>
concept CanEmplaceHint =
    requires(M m, typename M::const_iterator i, Args&&... args) { m.emplace_hint(i, std::forward<Args>(args)...); };

using Map = std::flat_map<Emplaceable, Emplaceable>;
static_assert(CanEmplaceHint<Map>);
static_assert(CanEmplaceHint<Map, Emplaceable, Emplaceable>);
static_assert(CanEmplaceHint<Map, std::piecewise_construct_t, std::tuple<int, double>, std::tuple<int, double>>);
static_assert(!CanEmplaceHint<Map, Emplaceable>);
static_assert(!CanEmplaceHint<Map, int, double>);
#endif

template <class KeyContainer, class ValueContainer>
void test_simple() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_map<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
  using R     = M::iterator;
  M m;
  ASSERT_SAME_TYPE(decltype(m.emplace_hint(m.cbegin())), R);
  R r = m.emplace_hint(m.end(), typename M::value_type(2, 3.5));
  assert(r == m.begin());
  assert(m.size() == 1);
  assert(m.begin()->first == 2);
  assert(m.begin()->second == 3.5);
}

template <class KeyContainer, class ValueContainer>
void test_emplaceable() {
  using M = std::flat_map<int, Emplaceable, std::less<int>, KeyContainer, ValueContainer>;
  using R = M::iterator;

  M m;
  ASSERT_SAME_TYPE(decltype(m.emplace_hint(m.cbegin())), R);
  R r = m.emplace_hint(m.end(), std::piecewise_construct, std::forward_as_tuple(2), std::forward_as_tuple());
  assert(r == m.begin());
  assert(m.size() == 1);
  assert(m.begin()->first == 2);
  assert(m.begin()->second == Emplaceable());
  r = m.emplace_hint(m.end(), std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(2, 3.5));
  assert(r == m.begin());
  assert(m.size() == 2);
  assert(m.begin()->first == 1);
  assert(m.begin()->second == Emplaceable(2, 3.5));
  r = m.emplace_hint(m.end(), std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(2, 3.5));
  assert(r == m.begin());
  assert(m.size() == 2);
  assert(m.begin()->first == 1);
  assert(m.begin()->second == Emplaceable(2, 3.5));
}

int main(int, char**) {
  test_simple<std::vector<int>, std::vector<double>>();
  test_simple<std::deque<int>, std::vector<double>>();
  test_simple<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test_simple<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  test_emplaceable<std::vector<int>, std::vector<Emplaceable>>();
  test_emplaceable<std::deque<int>, std::vector<Emplaceable>>();
  test_emplaceable<MinSequenceContainer<int>, MinSequenceContainer<Emplaceable>>();
  test_emplaceable<std::vector<int, min_allocator<int>>, std::vector<Emplaceable, min_allocator<Emplaceable>>>();

  {
    auto emplace_func = [](auto& m, auto key_arg, auto value_arg) {
      m.emplace_hint(m.begin(), std::piecewise_construct, std::tuple(key_arg), std::tuple(value_arg));
    };
    test_emplace_exception_guarantee(emplace_func);
  }

  return 0;
}
