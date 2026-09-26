//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// reverse_iterator       rbegin() noexcept;
// reverse_iterator       rend() noexcept;
// const_reverse_iterator rbegin() const noexcept;
// const_reverse_iterator rend() const noexcept;
// const_reverse_iterator crbegin() const noexcept;
// const_reverse_iterator crend() const noexcept;

#include <cassert>
#include <inplace_vector>
#include <iterator>

#include "test_macros.h"

constexpr bool test() {
  {
    using C = std::inplace_vector<int, 8>;
    C c;
    const C& cc = c;
    ASSERT_SAME_TYPE(C::reverse_iterator, decltype(c.rbegin()));
    ASSERT_SAME_TYPE(C::reverse_iterator, decltype(c.rend()));
    ASSERT_SAME_TYPE(C::const_reverse_iterator, decltype(cc.rbegin()));
    ASSERT_SAME_TYPE(C::const_reverse_iterator, decltype(cc.rend()));
    ASSERT_SAME_TYPE(C::const_reverse_iterator, decltype(c.crbegin()));
    ASSERT_SAME_TYPE(C::const_reverse_iterator, decltype(c.crend()));
    assert(c.rbegin() == c.rend());
    assert(c.crbegin() == c.crend());
  }
  {
    using C = std::inplace_vector<int, 8>;
    C c{1, 2, 3};
    C::reverse_iterator i = c.rbegin();
    assert(*i == 3);
    *i = 4;
    assert(c.back() == 4);
    ++i;
    assert(*i == 2);
    assert(std::distance(c.rbegin(), c.rend()) == 3);
  }
  {
    using C = std::inplace_vector<int, 8>;
    const C c{1, 2, 3};
    C::const_reverse_iterator i = c.rbegin();
    assert(*i == 3);
    assert(c.crbegin() == c.rbegin());
    assert(c.crend() == c.rend());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
