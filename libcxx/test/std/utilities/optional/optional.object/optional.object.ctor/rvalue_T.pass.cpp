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

// constexpr optional(T&& v);

#include <cassert>
#include <optional>
#include <type_traits>

#include "test_macros.h"
#include "archetypes.h"

using std::optional;

class Z {
public:
  Z(int) {}
  Z(Z&&) { TEST_THROW(6); }
};

template <typename T, typename U>
constexpr void test_rvalueT(U arg) {
  {
    const optional<T> opt(arg);
    assert(bool(opt));
    assert(*opt == T(arg));
  }

  {
    T t(arg);
    optional<T> opt(std::move(t));
    assert(*opt == T(arg));
  }

  struct test_constexpr_ctor : public optional<T> {
    constexpr test_constexpr_ctor(T&&) {}
    constexpr test_constexpr_ctor(const T&) {}
  };
}

void test_rt() {
  {
    typedef TestTypes::TestType T;
    T::reset();
    optional<T> opt = T{3};
    assert(T::alive == 1);
    assert(T::move_constructed == 1);
    assert(static_cast<bool>(opt) == true);
    assert(opt.value().value == 3);
  }
  {
    typedef ExplicitTestTypes::TestType T;
    static_assert(!std::is_convertible<T&&, optional<T>>::value, "");
    T::reset();
    optional<T> opt(T{3});
    assert(T::alive == 1);
    assert(T::move_constructed == 1);
    assert(static_cast<bool>(opt) == true);
    assert(opt.value().value == 3);
  }
  {
    typedef TestTypes::TestType T;
    T::reset();
    optional<T> opt = {3};
    assert(T::alive == 1);
    assert(T::value_constructed == 1);
    assert(T::copy_constructed == 0);
    assert(T::move_constructed == 0);
    assert(static_cast<bool>(opt) == true);
    assert(opt.value().value == 3);
  }
}

TEST_CONSTEXPR_CXX26 void test_throwing() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    try {
      Z z(3);
      optional<Z> opt(std::move(z));
      assert(false);
    } catch (int i) {
      assert(i == 6);
    }
  }
#endif
}

constexpr bool test() {
  test_rvalueT<int>(5);
  test_rvalueT<double>(3.0);
  test_rvalueT<const int>(42);

  {
    using T = ConstexprTestTypes::TestType;
    test_rvalueT<T>(T(3));
  }

  {
    using T = ConstexprTestTypes::TestType;
    test_rvalueT<T>(3);
  }

  {
    using T = ExplicitConstexprTestTypes::TestType;
    static_assert(!std::is_convertible<T&&, optional<T>>::value);
    test_rvalueT<T>(T(3));
  }
#if TEST_STD_VER >= 26 && 0
  {
    test_throwing();
  }

#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  {
    test_rt();
  }

  {
    test_throwing();
  }

  return 0;
}
