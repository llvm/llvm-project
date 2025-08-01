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

// iterator insert(const value_type& v);

#include <flat_map>
#include <deque>
#include <cassert>
#include <functional>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "../helpers.h"
#include "min_allocator.h"

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
  using R     = typename M::iterator;
  using VT    = typename M::value_type;
  M m;

  const VT v1(2, 2.5);
  std::same_as<R> decltype(auto) r = m.insert(v1);
  assert(r == m.begin());
  assert(m.size() == 1);
  assert(r->first == 2);
  assert(r->second == 2.5);

  const VT v2(1, 1.5);
  r = m.insert(v2);
  assert(r == m.begin());
  assert(m.size() == 2);
  assert(r->first == 1);
  assert(r->second == 1.5);

  const VT v3(3, 3.5);
  r = m.insert(v3);
  assert(r == m.begin() + 2);
  assert(m.size() == 3);
  assert(r->first == 3);
  assert(r->second == 3.5);

  const VT v4(3, 4.5);
  r = m.insert(v4);
  assert(r == m.begin() + 3);
  assert(m.size() == 4);
  assert(r->first == 3);
  assert(r->second == 4.5);
}

constexpr bool test() {
  test<std::vector<int>, std::vector<double>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  if (!TEST_IS_CONSTANT_EVALUATED) {
    auto insert_func = [](auto& m, auto key_arg, auto value_arg) {
      using FlatMap    = std::decay_t<decltype(m)>;
      using value_type = typename FlatMap::value_type;
      const value_type p(std::piecewise_construct, std::tuple(key_arg), std::tuple(value_arg));
      m.insert(p);
    };
    test_emplace_exception_guarantee(insert_func);
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
