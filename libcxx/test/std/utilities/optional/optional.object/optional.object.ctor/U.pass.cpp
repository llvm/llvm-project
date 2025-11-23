//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <optional>

// template <class U>
//   constexpr explicit optional(U&& u);

#include <cassert>
#include <optional>
#include <type_traits>

#include "archetypes.h"
#include "test_convertible.h"

using std::optional;

struct ImplicitThrow {
  constexpr ImplicitThrow(int x) {
    if (x != -1)
      TEST_THROW(6);
  }
};

struct ExplicitThrow {
  constexpr explicit ExplicitThrow(int x) {
    if (x != -1)
      TEST_THROW(6);
  }
};

struct ImplicitAny {
  template <class U>
  constexpr ImplicitAny(U&&) {}
};

template <class To, class From>
constexpr bool implicit_conversion(optional<To>&& opt, const From& v) {
  using O = optional<To>;
  assert((test_convertible<O, From>()));
  assert(!(test_convertible<O, void*>()));
  assert(!(test_convertible<O, From, int>()));
  assert(opt);
  assert(*opt == static_cast<To>(v));

  return true;
}

template <class To, class Input, class Expect>
constexpr bool explicit_conversion(Input&& in, const Expect& v) {
  using O = optional<To>;
  assert((std::is_constructible<O, Input>::value));
  assert(!(std::is_convertible<Input, O>::value));
  assert(!(std::is_constructible<O, void*>::value));
  assert(!(std::is_constructible<O, Input, int>::value));

  optional<To> opt(std::forward<Input>(in));
  optional<To> opt2{std::forward<Input>(in)};
  assert(opt);
  assert(opt2);
  assert(*opt == static_cast<To>(v));
  assert(*opt2 == static_cast<To>(v));

  return true;
}

void test_implicit() {
  {
    using T = TestTypes::TestType;
    optional<T> opt({3});
    assert(opt && *opt == static_cast<T>(3));
  }

  {
    using T = TestTypes::TestType;
    assert((implicit_conversion<T>(3, T(3))));
  }
}

void test_explicit() {
  {
    using T = ExplicitTestTypes::TestType;
    T::reset();
    {
      assert(explicit_conversion<T>(42, 42));
      assert(T::alive == 0);
    }
    T::reset();
    {
      optional<T> t(42);
      assert(T::alive == 1);
      assert(T::value_constructed == 1);
      assert(T::move_constructed == 0);
      assert(T::copy_constructed == 0);
      assert(t.value().value == 42);
    }
    assert(T::alive == 0);
  }
}

TEST_CONSTEXPR_CXX26 void test_throwing() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    try {
      using T       = ImplicitThrow;
      optional<T> t = 42;
      assert(false);
      ((void)t);
    } catch (int) {
    }
  }

  {
    try {
      using T = ExplicitThrow;
      optional<T> t(42);
      assert(false);
    } catch (int) {
    }
  }
#endif
}

constexpr bool test() {
  {
    assert((implicit_conversion<long long>(42, 42)));
  }

  {
    assert((implicit_conversion<long double>(3.14, 3.14)));
  }

  {
    int x = 42;
    optional<void* const> o(&x);
    assert(*o == &x);
  }

  {
    using T = TrivialTestTypes::TestType;
    assert((implicit_conversion<T>(42, 42)));
  }

  {
    using O = optional<ImplicitAny>;
    assert(!(test_convertible<O, std::in_place_t>()));
    assert(!(test_convertible<O, std::in_place_t&>()));
    assert(!(test_convertible<O, const std::in_place_t&>()));
    assert(!(test_convertible<O, std::in_place_t&&>()));
    assert(!(test_convertible<O, const std::in_place_t&&>()));
  }

  {
    using T = ExplicitTrivialTestTypes::TestType;
    assert((explicit_conversion<T>(42, 42)));
  }

  {
    using T = ExplicitConstexprTestTypes::TestType;
    assert(explicit_conversion<T>(42, 42));
    assert(!(std::is_convertible<int, T>::value));
  }

#if TEST_STD_VER >= 26 && 0
  test_throwing();
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  {
    test_implicit();
  }

  {
    test_explicit();
  }

  {
    test_throwing();
  }

  return 0;
}
