//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_multimap& operator=(flat_multimap&&);

#include <algorithm>
#include <deque>
#include <flat_map>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "test_macros.h"
#include "MoveOnly.h"
#include "../../../test_compare.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <template <class...> class KeyContainer, template <class...> class ValueContainer>
constexpr void test() {
  {
    using C  = test_less<int>;
    using A1 = test_allocator<int>;
    using A2 = test_allocator<char>;
    using M  = std::flat_multimap<int, char, C, KeyContainer<int, A1>, ValueContainer<char, A2>>;
    M mo     = M({{1, 1}, {1, 3}, {3, 2}}, C(5), A1(7));
    M m      = M({}, C(3), A1(7));
    m        = std::move(mo);
    assert((m == M{{1, 1}, {1, 3}, {3, 2}}));
    assert(m.key_comp() == C(5));
    auto [ks, vs] = std::move(m).extract();
    assert(ks.get_allocator() == A1(7));
    assert(vs.get_allocator() == A2(7));
    assert(mo.empty());
  }
  {
    using C  = test_less<int>;
    using A1 = other_allocator<int>;
    using A2 = other_allocator<char>;
    using M  = std::flat_multimap<int, char, C, KeyContainer<int, A1>, ValueContainer<char, A2>>;
    M mo     = M({{4, 5}, {4, 4}}, C(5), A1(7));
    M m      = M({{1, 1}, {2, 2}, {3, 3}, {4, 4}}, C(3), A1(7));
    m        = std::move(mo);
    assert((m == M{{4, 5}, {4, 4}}));
    assert(m.key_comp() == C(5));
    auto [ks, vs] = std::move(m).extract();
    assert(ks.get_allocator() == A1(7));
    assert(vs.get_allocator() == A2(7));
    assert(mo.empty());
  }
  {
    using A = min_allocator<int>;
    using M = std::flat_multimap<int, int, std::greater<int>, KeyContainer<int, A>, ValueContainer<int, A>>;
    M mo    = M({{5, 1}, {5, 2}, {3, 3}}, A());
    M m     = M({{4, 4}, {3, 3}, {2, 2}, {1, 1}}, A());
    m       = std::move(mo);
    assert((m == M{{5, 1}, {5, 2}, {3, 3}}));
    auto [ks, vs] = std::move(m).extract();
    assert(ks.get_allocator() == A());
    assert(vs.get_allocator() == A());
    assert(mo.empty());
  }
}

constexpr bool test() {
  test<std::vector, std::vector>();

#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test<std::deque, std::deque>();
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