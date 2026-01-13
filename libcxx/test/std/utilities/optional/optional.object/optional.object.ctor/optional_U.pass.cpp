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
//   optional(optional<U>&& rhs);

// optional<T&>: optional(optional<U>& rhs)

#include <cassert>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#if TEST_STD_VER >= 26
#  include "copy_move_types.h"
#endif

#include "../optional_helper_types.h"

using std::optional;

template <class T, class U>
TEST_CONSTEXPR_CXX20 void test(optional<U>&& rhs, bool is_going_to_throw = false) {
  bool rhs_engaged = static_cast<bool>(rhs);
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    optional<T> lhs = std::move(rhs);
    assert(is_going_to_throw == false);
    assert(static_cast<bool>(lhs) == rhs_engaged);
  } catch (int i) {
    assert(i == 6);
  }
#else
  if (is_going_to_throw)
    return;
  optional<T> lhs = std::move(rhs);
  assert(static_cast<bool>(lhs) == rhs_engaged);
#endif
}

class X {
  int i_;

public:
  TEST_CONSTEXPR_CXX20 X(int i) : i_(i) {}
  TEST_CONSTEXPR_CXX20 X(X&& x) : i_(std::exchange(x.i_, 0)) {}
  TEST_CONSTEXPR_CXX20 ~X() { i_ = 0; }
  friend constexpr bool operator==(const X& x, const X& y) { return x.i_ == y.i_; }
};

struct Z {
  Z(int) { TEST_THROW(6); }
};

template <class T, class U>
TEST_CONSTEXPR_CXX20 bool test_all() {
  {
    optional<T> rhs;
    test<U>(std::move(rhs));
  }
  {
    optional<T> rhs(short{3});
    test<U>(std::move(rhs));
  }
  return true;
}

#if TEST_STD_VER >= 26
constexpr bool test_ref() {
  // optional(optional<U>&)
  {
    int i = 1;
    std::optional<int&> o1{i};
    std::optional<int&> o2{o1};

    ASSERT_NOEXCEPT(std::optional<int&>(o2));
    assert(o2.has_value());
    assert(&(*o1) == &(*o2));
    assert(*o1 == i);
    assert(*o2 == i);
  }

  {
    std::optional<int&> o1;
    std::optional<int&> o2{o1};
    ASSERT_NOEXCEPT(std::optional<int&>(o2));
    assert(!o2.has_value());
  }

  {
    ReferenceConversion<int> t{1, 2};
    std::optional<ReferenceConversion<int>&> o1(t);
    std::optional<int&> o2(o1);
    ASSERT_NOEXCEPT(std::optional<int&>(o1));
    assert(o2.has_value());
    assert(&(*o2) == &t.lvalue);
    assert(*o2 == 1);
  }
  // optional(optional<U>&&)
  {
    int i = 1;
    std::optional<int&> o1{i};
    std::optional<int&> o2{std::move(o1)};

    // trivial move constructor should just copy the reference
    ASSERT_NOEXCEPT(std::optional<int&>(o2));
    assert(o2.has_value());
    assert(&(*o1) == &(*o2));
    assert(*o1 == i);
    assert(*o2 == i);
  }

  {
    std::optional<int&> o1;
    std::optional<int&> o2{std::move(o1)};
    ASSERT_NOEXCEPT(std::optional<int&>(o2));
    assert(!o2.has_value());
  }
  {
    TracedCopyMove t{};
    std::optional<TracedCopyMove&> o1{t};
    std::optional<TracedCopyMove> o2{std::move(o1)};
    assert(t.constMove == 0);
    assert(t.nonConstMove == 0);
  }

  {
    ReferenceConversion<int> t{1, 2};
    std::optional<ReferenceConversion<int>&> o1(t);
    std::optional<int&> o2(std::move(o1));
    ASSERT_NOEXCEPT(std::optional<int&>(o1));
    assert(o2.has_value());
    assert(&(*o2) == &t.lvalue);
    assert(*o2 == 1);
  }

  return true;
}
#endif

int main(int, char**) {
  test_all<short, int>();
  test_all<int, X>();
#if TEST_STD_VER > 17
  static_assert(test_all<short, int>());
  static_assert(test_all<int, X>());
#endif
  {
    optional<int> rhs;
    test<Z>(std::move(rhs));
  }
  {
    optional<int> rhs(3);
    test<Z>(std::move(rhs), true);
  }

  static_assert(!(std::is_constructible<optional<X>, optional<Z>>::value), "");

#if TEST_STD_VER >= 26
  assert(test_ref());
  static_assert(test_ref());
#endif
  return 0;
}
