//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// iterator       begin() noexcept;
// iterator       end() noexcept;
// const_iterator begin() const noexcept;
// const_iterator end() const noexcept;
// const_iterator cbegin() const noexcept;
// const_iterator cend() const noexcept;

#include <cassert>
#include <compare>
#include <concepts>
#include <inplace_vector>
#include <iterator>

#include "test_macros.h"

struct A {
  int first;
  int second;
};

constexpr bool test() {
  {
    using C = std::inplace_vector<int, 8>;
    C c;
    const C& cc = c;
    ASSERT_SAME_TYPE(C::iterator, decltype(c.begin()));
    ASSERT_SAME_TYPE(C::iterator, decltype(c.end()));
    ASSERT_SAME_TYPE(C::const_iterator, decltype(cc.begin()));
    ASSERT_SAME_TYPE(C::const_iterator, decltype(cc.end()));
    ASSERT_SAME_TYPE(C::const_iterator, decltype(c.cbegin()));
    ASSERT_SAME_TYPE(C::const_iterator, decltype(c.cend()));
    C::iterator i = c.begin();
    C::iterator j = c.end();
    assert(std::distance(i, j) == 0);
    assert(i == j);
  }
  {
    using C = std::inplace_vector<int, 8>;
    const C c;
    C::const_iterator i = c.begin();
    C::const_iterator j = c.end();
    assert(std::distance(i, j) == 0);
    assert(i == j);
  }
  {
    using C = std::inplace_vector<int, 8>;
    C c;
    C::const_iterator i = c.cbegin();
    C::const_iterator j = c.cend();
    assert(std::distance(i, j) == 0);
    assert(i == j);
    assert(i == c.end());
  }
  {
    using C = std::inplace_vector<int, 16>;
    C c{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    C::iterator i = c.begin();
    assert(*i == 0);
    ++i;
    assert(*i == 1);
    *i = 10;
    assert(*i == 10);
    assert(std::distance(c.begin(), c.end()) == 10);
  }
  {
    using C = std::inplace_vector<int, 8>;
    C::iterator i{};
    C::const_iterator j{};
    assert(i == i);
    assert(j == j);
  }
  {
    using C = std::inplace_vector<A, 4>;
    C c                 = {A{1, 2}};
    C::iterator i       = c.begin();
    i->first            = 3;
    C::const_iterator j = i;
    assert(j->first == 3);
  }
  {
    using C = std::inplace_vector<int, 8>;
    C::iterator i{};
    C::iterator j{};
    C::const_iterator ci{};

    assert(i == j);
    assert(i == ci);
    assert(ci == i);
    assert(!(i != ci));
    assert(!(ci != i));
    assert(!(i < ci));
    assert(!(ci < i));
    assert(i <= ci);
    assert(ci <= i);
    assert(!(i > ci));
    assert(!(ci > i));
    assert(i >= ci);
    assert(ci >= i);
    assert(ci - i == 0);
    assert(i - ci == 0);

    std::same_as<std::strong_ordering> decltype(auto) r1 = i <=> j;
    assert(r1 == std::strong_ordering::equal);
    std::same_as<std::strong_ordering> decltype(auto) r2 = ci <=> j;
    assert(r2 == std::strong_ordering::equal);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
