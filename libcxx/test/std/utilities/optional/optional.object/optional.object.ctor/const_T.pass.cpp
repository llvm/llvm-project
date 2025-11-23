//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14

// <optional>

// constexpr optional(const T& v);

#include <cassert>
#include <optional>
#include <type_traits>

#include "test_macros.h"
#include "archetypes.h"

using std::optional;
template <typename T, typename... Args>
constexpr void test_ctor(Args... args) {
  const T t(args...);
  const std::optional<T> opt(t);

  assert(static_cast<bool>(opt));
  assert(*opt == t);

  struct test_constexpr_ctor : public optional<T> {
    constexpr test_constexpr_ctor(const T&) {}
  };
}

constexpr bool test() {
  test_ctor<int>(5);
  test_ctor<double>(3.0);
  test_ctor<const int>(42);
  test_ctor<ConstexprTestTypes::TestType>(3);

  {
    using T = ExplicitConstexprTestTypes::TestType;
    static_assert(!std::is_convertible_v<const T&, optional<T>>);
    test_ctor<T>(3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  {
    typedef TestTypes::TestType T;
    T::reset();
    const T t(3);
    optional<T> opt = t;
    assert(T::alive == 2);
    assert(T::copy_constructed == 1);
    assert(static_cast<bool>(opt) == true);
    assert(opt.value().value == 3);
  }

  {
    typedef ExplicitTestTypes::TestType T;
    static_assert(!std::is_convertible<T const&, optional<T>>::value, "");
    T::reset();
    const T t(3);
    optional<T> opt(t);
    assert(T::alive == 2);
    assert(T::copy_constructed == 1);
    assert(static_cast<bool>(opt) == true);
    assert(opt.value().value == 3);
  }

#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    struct Z {
      Z(int) {}
      Z(const Z&) { throw 6; }
    };
    typedef Z T;
    try {
      const T t(3);
      optional<T> opt(t);
      assert(false);
    } catch (int i) {
      assert(i == 6);
    }
  }
#endif

  return 0;
}
