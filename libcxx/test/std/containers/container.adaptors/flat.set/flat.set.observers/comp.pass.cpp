//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// key_compare key_comp() const;
// value_compare value_comp() const;

#include <cassert>
#include <flat_set>
#include <functional>
#include <utility>
#include <vector>

#include "test_macros.h"

int main(int, char**) {
  {
    using M    = std::flat_set<int>;
    using Comp = std::less<int>; // the default
    M m        = {};
    ASSERT_SAME_TYPE(M::key_compare, Comp);
    ASSERT_SAME_TYPE(decltype(m.key_comp()), Comp);
    ASSERT_SAME_TYPE(decltype(m.value_comp()), Comp);
    Comp kc = m.key_comp();
    assert(kc(1, 2));
    assert(!kc(2, 1));
    auto vc = m.value_comp();
    assert(vc(1, 2));
    assert(!vc(2, 1));
  }
  {
    using Comp = std::function<bool(int, int)>;
    using M    = std::flat_set<int, Comp>;
    Comp comp  = std::greater<int>();
    M m({}, comp);
    ASSERT_SAME_TYPE(M::key_compare, Comp);
    ASSERT_SAME_TYPE(decltype(m.key_comp()), Comp);
    ASSERT_SAME_TYPE(decltype(m.value_comp()), Comp);
    Comp kc = m.key_comp();
    assert(!kc(1, 2));
    assert(kc(2, 1));
    auto vc = m.value_comp();
    assert(!vc(1, 2));
    assert(vc(2, 1));
  }
  {
    using Comp = std::less<>;
    using M    = std::flat_set<int, Comp>;
    M m        = {};
    ASSERT_SAME_TYPE(M::key_compare, Comp);
    ASSERT_SAME_TYPE(decltype(m.key_comp()), Comp);
    ASSERT_SAME_TYPE(decltype(m.value_comp()), Comp);
    Comp kc = m.key_comp();
    assert(kc(1, 2));
    assert(!kc(2, 1));
    auto vc = m.value_comp();
    auto a  = std::make_pair(1, 2);
    ASSERT_SAME_TYPE(decltype(vc(a, a)), bool);
    assert(vc(1, 2));
    assert(!vc(2, 1));
  }

  return 0;
}
