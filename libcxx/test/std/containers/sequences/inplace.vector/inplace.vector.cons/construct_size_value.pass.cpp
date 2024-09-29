//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// inplace_vector(size_type n, const value_type& x);

#include <inplace_vector>
#include <cassert>

#include "test_macros.h"

template <class C>
constexpr void test(typename C::size_type n, const typename C::value_type& x) {
  C c(n, x);
  assert(c.size() == n);
  for (typename C::const_iterator i = c.cbegin(), e = c.cend(); i != e; ++i)
    assert(*i == x);
}

constexpr bool tests() {
  test<std::inplace_vector<int, 0>>(0, 3);
  test<std::inplace_vector<int, 10>>(0, 3);
  test<std::inplace_vector<int, 100>>(50, 3);
  test<std::inplace_vector<int, 100>>(100, 3);
  if !consteval {
#ifndef TEST_HAS_NO_EXCEPTIONS
    try {
      std::inplace_vector<int, 10>(50, 3);
      assert(false);
    } catch (const std::bad_alloc&) {
      // OK
    } catch (...) {
      assert(false);
    }

    try {
      std::inplace_vector<int, 0>(1, 3);
      assert(false);
    } catch (const std::bad_alloc&) {
      // OK
    } catch (...) {
      assert(false);
    }
#endif
  }
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
