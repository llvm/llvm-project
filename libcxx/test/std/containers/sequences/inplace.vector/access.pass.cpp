//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

//       reference operator[](size_type __i);
// const_reference operator[](size_type __i) const;
//
//       reference at(size_type __i);
// const_reference at(size_type __i) const;
//
//       reference front();
// const_reference front() const;
//
//       reference back();
// const_reference back() const;
// libc++ marks these as 'noexcept' (except 'at')

#include <inplace_vector>
#include <cassert>
#include <stdexcept>

#include "test_macros.h"
#include "MoveOnly.h"

template <class C>
constexpr C make(int size, int start) {
  C c;
  for (int i = 0; i < size; ++i)
    c.push_back(start + i);
  return c;
}

template <class Vector>
constexpr void test_get_basic(Vector& c, int start_value) {
  const int n = static_cast<int>(c.size());
  for (int i = 0; i < n; ++i)
    assert(c[i] == start_value + i);
  for (int i = 0; i < n; ++i)
    assert(c.at(i) == start_value + i);

#ifndef TEST_HAS_NO_EXCEPTIONS
  if !consteval {
    try {
      TEST_IGNORE_NODISCARD c.at(n);
      assert(false);
    } catch (const std::out_of_range&) {
    } catch (...) {
      assert(false);
    }
  }
#endif

  assert(c.front() == start_value);
  assert(c.back() == start_value + n - 1);
}

template <class Vector, bool IsConstexpr>
constexpr void test_get() {
  bool can_run = Vector::capacity() >= 10;
  if consteval {
    can_run &= IsConstexpr;
  }
  if (can_run) {
    int start_value  = 35;
    Vector c         = make<Vector>(10, start_value);
    const Vector& cc = c;
    test_get_basic(c, start_value);
    test_get_basic(cc, start_value);
  }
}

template <class Vector, bool IsConstexpr>
constexpr void test_set() {
  const int n  = 10;
  bool can_run = Vector::capacity() >= n;
  if consteval {
    can_run &= IsConstexpr;
  }
  if (can_run) {
    int start_value = 35;
    Vector c        = make<Vector>(n, start_value);

    for (int i = 0; i < n; ++i) {
      assert(c[i] == start_value + i);
      c[i] = start_value + i + 1;
      assert(c[i] == start_value + i + 1);
    }
    for (int i = 0; i < n; ++i) {
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
}

template <class Vector, bool IsConstexpr>
constexpr void test() {
  test_get<Vector, IsConstexpr>();
  test_set<Vector, IsConstexpr>();

  union {
    int _ = 0;
    Vector c;
  };
  const Vector& cc = c;
  ASSERT_SAME_TYPE(typename Vector::reference, decltype(c[0]));
  ASSERT_SAME_TYPE(typename Vector::const_reference, decltype(cc[0]));

  ASSERT_SAME_TYPE(typename Vector::reference, decltype(c.at(0)));
  ASSERT_SAME_TYPE(typename Vector::const_reference, decltype(cc.at(0)));

  ASSERT_SAME_TYPE(typename Vector::reference, decltype(c.front()));
  ASSERT_SAME_TYPE(typename Vector::const_reference, decltype(cc.front()));

  ASSERT_SAME_TYPE(typename Vector::reference, decltype(c.back()));
  ASSERT_SAME_TYPE(typename Vector::const_reference, decltype(cc.back()));
}

constexpr bool tests() {
  test<std::inplace_vector<int, 0>, true>();
  test<std::inplace_vector<int, 10>, true>();
  test<std::inplace_vector<int, 100>, true>();
  test<std::inplace_vector<MoveOnly, 0>, true>();
  test<std::inplace_vector<MoveOnly, 10>, false>();
  test<std::inplace_vector<MoveOnly, 100>, false>();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
