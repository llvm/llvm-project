//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class I2, class S2>
//   requires convertible_to<const I2&, I> && convertible_to<const S2&, S>
//     constexpr common_iterator(const common_iterator<I2, S2>& x);

#include <iterator>
#include <cassert>

#include "test_macros.h"

constexpr bool test()
{
  struct Base {};
  struct Derived : Base {};

  using BaseIt = std::common_iterator<Base*, const Base*>;
  using DerivedIt = std::common_iterator<Derived*, const Derived*>;
  static_assert(std::is_convertible_v<DerivedIt, BaseIt>); // Derived* to Base*
  static_assert(!std::is_constructible_v<DerivedIt, BaseIt>); // Base* to Derived*

  Derived a[10] = {};
  DerivedIt it = DerivedIt(a); // the iterator type
  BaseIt jt = BaseIt(it);
  assert(jt == BaseIt(a));

  it = DerivedIt((const Derived*)a); // the sentinel type
  jt = BaseIt(it);
  assert(jt == BaseIt(a));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
