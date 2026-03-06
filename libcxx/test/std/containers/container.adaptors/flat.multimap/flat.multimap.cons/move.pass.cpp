//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_multimap(flat_multimap&&);

#include <algorithm>
#include <deque>
#include <flat_map>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include "../helpers.h"
#include "test_macros.h"
#include "../../../test_compare.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <template <class...> class KeyContainer, template <class...> class ValueContainer>
constexpr void test() {
  {
    using C = test_less<int>;
    using A = test_allocator<int>;
    using M = std::flat_multimap<int, int, C, KeyContainer<int, A>, ValueContainer<int, A>>;
    M mo    = M({{1, 1}, {1, 2}, {3, 1}}, C(5), A(7));
    M m     = std::move(mo);
    assert((m == M{{1, 1}, {1, 2}, {3, 1}}));
    assert(m.key_comp() == C(5));
    assert(m.keys().get_allocator() == A(7));
    assert(m.values().get_allocator() == A(7));

    assert(mo.empty());
    assert(mo.key_comp() == C(5));
    assert(mo.keys().get_allocator().get_id() == test_alloc_base::moved_value);
    assert(mo.values().get_allocator().get_id() == test_alloc_base::moved_value);
  }
  {
    using C = test_less<int>;
    using A = min_allocator<int>;
    using M = std::flat_multimap<int, int, C, KeyContainer<int, A>, ValueContainer<int, A>>;
    M mo    = M({{1, 1}, {1, 2}, {3, 1}}, C(5), A());
    M m     = std::move(mo);
    assert((m == M{{1, 1}, {1, 2}, {3, 1}}));
    assert(m.key_comp() == C(5));
    assert(m.keys().get_allocator() == A());
    assert(m.values().get_allocator() == A());

    assert(mo.empty());
    assert(mo.key_comp() == C(5));
    assert(m.keys().get_allocator() == A());
    assert(m.values().get_allocator() == A());
  }
  if (!TEST_IS_CONSTANT_EVALUATED) {
    // A moved-from flat_multimap maintains its class invariant in the presence of moved-from comparators.
    using M = std::flat_multimap<int, int, std::function<bool(int, int)>, KeyContainer<int>, ValueContainer<int>>;
    M mo    = M({{1, 1}, {1, 2}, {3, 1}}, std::less<int>());
    M m     = std::move(mo);
    assert(m.size() == 3);
    assert(std::is_sorted(m.begin(), m.end(), m.value_comp()));
    assert(m.key_comp()(1, 2) == true);

    assert(std::is_sorted(mo.begin(), mo.end(), mo.value_comp()));
    LIBCPP_ASSERT(m.key_comp()(1, 2) == true);
    LIBCPP_ASSERT(mo.empty());
    mo.insert({{1, 1}, {1, 2}, {3, 1}}); // insert has no preconditions
    assert(m == mo);
  }
  {
    // moved-from object maintains invariant if one of underlying container does not clear after move
    using M = std::flat_multimap<int, int, std::less<>, KeyContainer<int>, CopyOnlyVector<int>>;
    M m1    = M({1, 1, 3}, {1, 2, 3});
    M m2    = std::move(m1);
    assert(m2.size() == 3);
    check_invariant(m1);
    LIBCPP_ASSERT(m1.empty());
    LIBCPP_ASSERT(m1.keys().size() == 0);
    LIBCPP_ASSERT(m1.values().size() == 0);
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
