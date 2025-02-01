//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

//       reverse_iterator rbegin() noexcept;
// const_reverse_iterator rbegin() const noexcept;
//       reverse_iterator rend()   noexcept;
// const_reverse_iterator rend()   const noexcept;
//
// const_reverse_iterator crbegin() const noexcept;
// const_reverse_iterator crend()   const noexcept;

#include <cassert>
#include <cstddef>
#include <deque>
#include <flat_set>
#include <functional>
#include <vector>

#include <iterator>

#include "test_macros.h"
#include <iostream>

int main(int, char**) {
  {
    using M     = std::flat_set<int, std::less<int>, std::deque<int>>;
    M m         = {1, 2, 3, 4};
    const M& cm = m;
    ASSERT_SAME_TYPE(decltype(m.rbegin()), M::reverse_iterator);
    ASSERT_SAME_TYPE(decltype(m.crbegin()), M::const_reverse_iterator);
    ASSERT_SAME_TYPE(decltype(cm.rbegin()), M::const_reverse_iterator);
    ASSERT_SAME_TYPE(decltype(m.rend()), M::reverse_iterator);
    ASSERT_SAME_TYPE(decltype(m.crend()), M::const_reverse_iterator);
    ASSERT_SAME_TYPE(decltype(cm.rend()), M::const_reverse_iterator);
    static_assert(noexcept(m.rbegin()));
    static_assert(noexcept(cm.rbegin()));
    static_assert(noexcept(m.crbegin()));
    static_assert(noexcept(m.rend()));
    static_assert(noexcept(cm.rend()));
    static_assert(noexcept(m.crend()));
    assert(m.size() == 4);
    assert(std::distance(m.rbegin(), m.rend()) == 4);
    assert(std::distance(cm.rbegin(), cm.rend()) == 4);
    assert(std::distance(m.crbegin(), m.crend()) == 4);
    assert(std::distance(cm.crbegin(), cm.crend()) == 4);
    M::reverse_iterator i; // default-construct
    ASSERT_SAME_TYPE(decltype(*i), const int&);
    i                           = m.rbegin(); // move-assignment
    M::const_reverse_iterator k = i;          // converting constructor
    assert(i == k);                           // comparison
    for (int j = 4; j >= 1; --j, ++i) {       // pre-increment
      assert(*i == j);
    }
    assert(i == m.rend());
    for (int j = 1; j <= 4; ++j) {
      --i; // pre-decrement
      assert(*i == j);
    }
    assert(i == m.rbegin());
  }
  {
    // N3644 testing
    using C = std::flat_set<int, char>;
    C::reverse_iterator ii1{}, ii2{};
    C::reverse_iterator ii4 = ii1;
    C::const_reverse_iterator cii{};
    assert(ii1 == ii2);
    assert(ii1 == ii4);
    assert(!(ii1 != ii2));

    assert((ii1 == cii));
    assert((cii == ii1));
    assert(!(ii1 != cii));
    assert(!(cii != ii1));
  }

  return 0;
}
