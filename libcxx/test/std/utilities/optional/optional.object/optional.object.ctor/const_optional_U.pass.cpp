//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// template <class U>
//   optional(const optional<U>& rhs);
//optional<T&>: optional(const optional<U>&& rhs);

#include <cassert>
#include <optional>
#include <type_traits>

#include "../optional_helper_types.h"
#include "test_macros.h"

using std::optional;

template <class T, class U>
TEST_CONSTEXPR_CXX20 void test(const optional<U>& rhs, bool is_going_to_throw = false) {
  bool rhs_engaged = static_cast<bool>(rhs);
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    optional<T> lhs = rhs;
    assert(is_going_to_throw == false);
    assert(static_cast<bool>(lhs) == rhs_engaged);
    if (rhs_engaged)
      assert(*lhs == *rhs);
  } catch (int i) {
    assert(i == 6);
  }
#else
  if (is_going_to_throw)
    return;
  optional<T> lhs = rhs;
  assert(static_cast<bool>(lhs) == rhs_engaged);
  if (rhs_engaged)
    assert(*lhs == *rhs);
#endif
}

class X {
  int i_;

public:
  constexpr X(int i) : i_(i) {}
  constexpr X(const X& x) : i_(x.i_) {}
  TEST_CONSTEXPR_CXX20 ~X() { i_ = 0; }
  friend constexpr bool operator==(const X& x, const X& y) { return x.i_ == y.i_; }
};

class Y {
  int i_;

public:
  constexpr Y(int i) : i_(i) {}

  friend constexpr bool operator==(const Y& x, const Y& y) { return x.i_ == y.i_; }
};

int count = 0;

class Z {
  int i_;

public:
  Z(int i) : i_(i) { TEST_THROW(6); }

  friend bool operator==(const Z& x, const Z& y) { return x.i_ == y.i_; }
};

template <class T, class U>
constexpr bool test_all() {
  {
    optional<U> rhs;
    test<T>(rhs);
  }
  {
    optional<U> rhs(U{3});
    test<T>(rhs);
  }
  return true;
}

#if TEST_STD_VER >= 26
constexpr bool test_ref() {
  // optional(const optional<U>&)
  {
    int i = 1;
    const std::optional<int&> o1{i};
    const std::optional<int&> o2{o1};

    ASSERT_NOEXCEPT(std::optional<int&>(o2));
    assert(o2.has_value());
    assert(&(*o1) == &(*o2));
    assert(*o1 == i);
    assert(*o2 == i);
  }

  {
    const std::optional<int&> o1;
    const std::optional<int&> o2{o1};
    ASSERT_NOEXCEPT(std::optional<int&>(o2));
    assert(!o2.has_value());
  }

  {
    ReferenceConversion<int> t{1, 2};
    const std::optional<ReferenceConversion<int>&> o1(t);
    const std::optional<int&> o2(o1);
    ASSERT_NOEXCEPT(std::optional<int&>(o1));
    assert(o2.has_value());
    assert(&(*o2) == &t.lvalue);
    assert(*o2 == 1);
  }

  // optional(const optional<U>&&)
  {
    int i = 1;
    const std::optional<int&> o1{i};
    const std::optional<int&> o2{std::move(o1)};

    // trivial move constructor should just copy the reference
    ASSERT_NOEXCEPT(std::optional<int&>(std::move(o2)));
    assert(o2.has_value());
    assert(&(*o1) == &(*o2));
    assert(*o1 == i);
    assert(*o2 == i);
  }

  {
    const std::optional<int&> o1;
    const std::optional<int&> o2{std::move(o1)};
    ASSERT_NOEXCEPT(std::optional<int&>(o2));
    assert(!o2.has_value());
  }

  {
    ReferenceConversion<int> t{1, 2};
    const std::optional<ReferenceConversion<int>&> o1(t);
    const std::optional<int&> o2(std::move(o1));
    ASSERT_NOEXCEPT(std::optional<int&>(o1));
    assert(o2.has_value());
    assert(&(*o2) == &t.lvalue);
    assert(*o2 == 1);
  }

  return true;
}
#endif

int main(int, char**) {
  test_all<int, short>();
  test_all<X, int>();
  test_all<Y, int>();
#if TEST_STD_VER > 17
  static_assert(test_all<int, short>());
  static_assert(test_all<X, int>());
  static_assert(test_all<Y, int>());
#endif
  {
    typedef Z T;
    typedef int U;
    optional<U> rhs;
    test<T>(rhs);
  }
  {
    typedef Z T;
    typedef int U;
    optional<U> rhs(U{3});
    test<T>(rhs, true);
  }

  static_assert(!(std::is_constructible<optional<X>, const optional<Y>&>::value), "");
#if TEST_STD_VER >= 26
  assert(test_ref());
  static_assert(test_ref());
#endif
  return 0;
}
