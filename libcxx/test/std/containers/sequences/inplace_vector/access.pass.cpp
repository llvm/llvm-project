//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

//       reference operator[](size_type n);
// const_reference operator[](size_type n) const;
//       reference at(size_type n);
// const_reference at(size_type n) const;
//       reference front();
// const_reference front() const;
//       reference back();
// const_reference back() const;

#include <cassert>
#include <inplace_vector>
#include <stdexcept>
#include <type_traits>

#include "test_macros.h"

template <class C>
constexpr C make(int size, int start) {
  C c;
  for (int i = 0; i != size; ++i)
    c.push_back(start + i);
  return c;
}

template <class C>
constexpr void test_get_basic(C& c, int start_value) {
  const int n = static_cast<int>(c.size());
  for (int i = 0; i != n; ++i)
    assert(c[i] == start_value + i);
  for (int i = 0; i != n; ++i)
    assert(c.at(i) == start_value + i);

#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) {
    try {
      (void)c.at(n);
      assert(false);
    } catch (const std::out_of_range&) {
    }
  }
#endif

  assert(c.front() == start_value);
  assert(c.back() == start_value + n - 1);
}

template <class C>
constexpr void test_get() {
  int start_value = 35;
  C c             = make<C>(10, start_value);
  const C& cc     = c;
  test_get_basic(c, start_value);
  test_get_basic(cc, start_value);
}

template <class C>
constexpr void test_set() {
  int start_value = 35;
  const int n     = 10;
  C c             = make<C>(n, start_value);

  for (int i = 0; i != n; ++i) {
    assert(c[i] == start_value + i);
    c[i] = start_value + i + 1;
    assert(c[i] == start_value + i + 1);
  }
  for (int i = 0; i != n; ++i) {
    assert(c.at(i) == start_value + i + 1);
    c.at(i) = start_value + i + 2;
    assert(c.at(i) == start_value + i + 2);
  }

  assert(c.front() == start_value + 2);
  c.front() = start_value + 3;
  assert(c.front() == start_value + 3);

  assert(c.back() == start_value + n + 1);
  c.back() = start_value + n + 2;
  assert(c.back() == start_value + n + 2);
}

template <class C>
constexpr void test() {
  test_get<C>();
  test_set<C>();

  C c;
  const C& cc = c;
  ASSERT_SAME_TYPE(typename C::reference, decltype(c[0]));
  ASSERT_SAME_TYPE(typename C::const_reference, decltype(cc[0]));
  ASSERT_SAME_TYPE(typename C::reference, decltype(c.at(0)));
  ASSERT_SAME_TYPE(typename C::const_reference, decltype(cc.at(0)));
  ASSERT_SAME_TYPE(typename C::reference, decltype(c.front()));
  ASSERT_SAME_TYPE(typename C::const_reference, decltype(cc.front()));
  ASSERT_SAME_TYPE(typename C::reference, decltype(c.back()));
  ASSERT_SAME_TYPE(typename C::const_reference, decltype(cc.back()));
}

constexpr bool tests() {
  test<std::inplace_vector<int, 16> >();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
