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

// key_compare key_comp() const;
// value_compare value_comp() const;

#include <cassert>
#include <flat_map>
#include <functional>
#include <utility>
#include <vector>

#include "test_macros.h"

constexpr bool test() {
  {
    using M    = std::flat_multimap<int, char>;
    using Comp = std::less<int>; // the default
    M m        = {};
    ASSERT_SAME_TYPE(M::key_compare, Comp);
    static_assert(!std::is_same_v<M::value_compare, Comp>);
    ASSERT_SAME_TYPE(decltype(m.key_comp()), Comp);
    ASSERT_SAME_TYPE(decltype(m.value_comp()), M::value_compare);
    Comp kc = m.key_comp();
    assert(kc(1, 2));
    assert(!kc(2, 1));
    auto vc = m.value_comp();
    ASSERT_SAME_TYPE(decltype(vc(std::make_pair(1, 2), std::make_pair(1, 2))), bool);
    assert(vc({1, '2'}, {2, '1'}));
    assert(!vc({2, '1'}, {1, '2'}));
  }
  if (!TEST_IS_CONSTANT_EVALUATED) {
    using Comp = std::function<bool(int, int)>;
    using M    = std::flat_multimap<int, int, Comp>;
    Comp comp  = std::greater<int>();
    M m({}, comp);
    ASSERT_SAME_TYPE(M::key_compare, Comp);
    ASSERT_SAME_TYPE(decltype(m.key_comp()), Comp);
    ASSERT_SAME_TYPE(decltype(m.value_comp()), M::value_compare);
    Comp kc = m.key_comp();
    assert(!kc(1, 2));
    assert(kc(2, 1));
    auto vc = m.value_comp();
    auto a  = std::make_pair(1, 2);
    ASSERT_SAME_TYPE(decltype(vc(a, a)), bool);
    static_assert(!noexcept(vc(a, a)));
    assert(!vc({1, 2}, {2, 1}));
    assert(vc({2, 1}, {1, 2}));
  }
  {
    using Comp = std::less<>;
    using M    = std::flat_multimap<int, int, Comp>;
    M m        = {};
    ASSERT_SAME_TYPE(M::key_compare, Comp);
    ASSERT_SAME_TYPE(decltype(m.key_comp()), Comp);
    ASSERT_SAME_TYPE(decltype(m.value_comp()), M::value_compare);
    Comp kc = m.key_comp();
    assert(kc(1, 2));
    assert(!kc(2, 1));
    auto vc = m.value_comp();
    auto a  = std::make_pair(1, 2);
    ASSERT_SAME_TYPE(decltype(vc(a, a)), bool);
    assert(vc({1, 2}, {2, 1}));
    assert(!vc({2, 1}, {1, 2}));
  }
  if (!TEST_IS_CONSTANT_EVALUATED) {
    using Comp = std::function<bool(const std::vector<int>&, const std::vector<int>&)>;
    using M    = std::flat_multimap<std::vector<int>, int, Comp>;
    Comp comp  = [i = 1](const auto& x, const auto& y) { return x[i] < y[i]; };
    M m({}, comp);
    auto vc = m.value_comp();
    static_assert(sizeof(vc) >= sizeof(Comp));
    comp = nullptr;
    m    = M({}, nullptr);
    assert(m.key_comp() == nullptr);
    // At this point, m.key_comp() is disengaged.
    // But the std::function captured by copy inside `vc` remains valid.
    auto a = std::make_pair(std::vector<int>{2, 1, 4}, 42);
    auto b = std::make_pair(std::vector<int>{1, 2, 3}, 42);
    auto c = std::make_pair(std::vector<int>{0, 3, 2}, 42);
    assert(vc(a, b));
    assert(vc(b, c));
    assert(!vc(b, a));
    assert(!vc(c, b));
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
