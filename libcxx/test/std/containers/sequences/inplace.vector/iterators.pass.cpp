//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// iterator       begin()        noexcept;
// iterator       end()          noexcept;
// const_iterator begin()  const noexcept;
// const_iterator end()    const noexcept;
// const_iterator cbegin() const noexcept;
// const_iterator cend()   const noexcept;

#include <inplace_vector>
#include <cassert>
#include <iterator>

#include "test_macros.h"

struct A {
  int first;
  int second;
};

constexpr bool tests() {
  {
    using C = std::inplace_vector<int, 0>;
    C c;
    std::same_as<C::iterator> decltype(auto) i = c.begin();
    C::iterator j                              = c.end();
    assert(std::distance(i, j) == 0);
    assert(i == j);
  }
  {
    using C = std::inplace_vector<int, 10>;
    C c;
    std::same_as<C::iterator> decltype(auto) i = c.begin();
    C::iterator j                              = c.end();
    assert(std::distance(i, j) == 0);
    assert(i == j);
  }
  {
    using C = std::inplace_vector<int, 0>;
    const C c;
    std::same_as<C::const_iterator> decltype(auto) i = c.begin();
    C::const_iterator j                              = c.end();
    assert(std::distance(i, j) == 0);
    assert(i == j);
  }
  {
    using C = std::inplace_vector<int, 10>;
    const C c;
    std::same_as<C::const_iterator> decltype(auto) i = c.begin();
    C::const_iterator j                              = c.end();
    assert(std::distance(i, j) == 0);
    assert(i == j);
  }
  {
    using C = std::inplace_vector<int, 10>;
    C c;
    std::same_as<C::const_iterator> decltype(auto) i = c.cbegin();
    C::const_iterator j                              = c.cend();
    assert(std::distance(i, j) == 0);
    assert(i == j);
    assert(i == c.end());
  }
  {
    using C       = std::inplace_vector<int, 10>;
    const int t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    C c(std::begin(t), std::end(t));
    C::iterator i = c.begin();
    assert(*i == 0);
    ++i;
    assert(*i == 1);
    *i = 10;
    assert(*i == 10);
    assert(std::distance(c.begin(), c.end()) == 10);
  }
  {
    using C = std::inplace_vector<int, 10>;
    [[maybe_unused]] C::iterator i;
    [[maybe_unused]] C::const_iterator j;
    C::iterator i2{};
    C::iterator i3 = i2;
    C::const_iterator j2{};
    C::const_iterator j3 = i2;
    C::const_iterator j4 = j2;
    assert(i2 == j2);
    j4 = j3;
    j2 = i2;
    i2 = i3;
  }
  {
    using C = std::inplace_vector<int, 10>;
    C c;
    C::iterator i = c.begin();
    C::iterator j = c.end();
    assert(std::distance(i, j) == 0);

    assert(i == j);
    assert(!(i != j));

    assert(!(i < j));
    assert((i <= j));

    assert(!(i > j));
    assert((i >= j));

    std::same_as<std::strong_ordering> decltype(auto) r1 = i <=> j;
    assert(r1 == std::strong_ordering::equal);
  }
  {
    using C = std::inplace_vector<int, 10>;
    const C c;
    C::const_iterator i = c.begin();
    C::const_iterator j = c.end();
    assert(std::distance(i, j) == 0);

    assert(i == j);
    assert(!(i != j));

    assert(!(i < j));
    assert((i <= j));

    assert(!(i > j));
    assert((i >= j));

    std::same_as<std::strong_ordering> decltype(auto) r1 = i <=> j;
    assert(r1 == std::strong_ordering::equal);
  }
  {
    using C       = std::inplace_vector<int, 10>;
    const int t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    C c(std::begin(t), std::end(t));
    C::iterator i = c.begin();
    C::iterator j = i;
    assert(*i == 0);
    ++i;
    assert(*i == 1);
    *i = 10;
    assert(*i == 10);
    assert(j == c.begin());
    assert(++j == i);
    assert(std::distance(c.begin(), c.end()) == 10);
  }
  {
    using C             = std::inplace_vector<A, 10>;
    C c                 = {A{1, 2}};
    C::iterator i       = c.begin();
    i->first            = 3;
    C::const_iterator j = i;
    assert(j->first == 3);
  }
  {
    using C = std::inplace_vector<int, 10>;
    C::iterator ii1{}, ii2{};
    C::iterator ii4 = ii1;
    C::const_iterator cii{};
    assert(ii1 == ii2);
    assert(ii1 == ii4);

    assert(!(ii1 != ii2));

    assert((ii1 == cii));
    assert((cii == ii1));
    assert(!(ii1 != cii));
    assert(!(cii != ii1));
    assert(!(ii1 < cii));
    assert(!(cii < ii1));
    assert((ii1 <= cii));
    assert((cii <= ii1));
    assert(!(ii1 > cii));
    assert(!(cii > ii1));
    assert((ii1 >= cii));
    assert((cii >= ii1));
    assert(cii - ii1 == 0);
    assert(ii1 - cii == 0);

    std::same_as<std::strong_ordering> decltype(auto) r1 = ii1 <=> ii2;
    assert(r1 == std::strong_ordering::equal);

    std::same_as<std::strong_ordering> decltype(auto) r2 = cii <=> ii2;
    assert(r2 == std::strong_ordering::equal);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
