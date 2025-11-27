//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <optional>

// constexpr optional(const optional<T>& rhs);

#include <cassert>
#include <optional>
#include <type_traits>

#include "test_macros.h"
#include "archetypes.h"

using std::optional;

template <class T, class... InitArgs>
constexpr bool test(InitArgs&&... args) {
  static_assert(std::is_trivially_copy_constructible_v<T> ==
                std::is_trivially_copy_constructible_v<std::optional<T>>); // requirement
  const optional<T> lhs(std::forward<InitArgs>(args)...);
  optional<T> rhs(lhs);
  assert(lhs.has_value() == rhs.has_value());
  assert(lhs.has_value() ? *lhs == *rhs : true);

  return true;
}

TEST_CONSTEXPR_CXX26 bool test_throwing_ctor() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Z {
    Z() : count(0) {}
    Z(Z const& o) : count(o.count + 1) {
      if (count == 2)
        throw 6;
    }
    int count;
  };
  const Z z;
  const optional<Z> rhs(z);
  try {
    optional<Z> lhs(rhs);
    assert(false);
  } catch (int i) {
    assert(i == 6);
  }
#endif

  return true;
}

template <class T, class... InitArgs>
void test_ref(InitArgs&&... args) {
  const optional<T> rhs(std::forward<InitArgs>(args)...);
  bool rhs_engaged = static_cast<bool>(rhs);
  optional<T> lhs  = rhs;
  assert(static_cast<bool>(lhs) == rhs_engaged);
  if (rhs_engaged)
    assert(&(*lhs) == &(*rhs));
}

#if TEST_STD_VER >= 26
struct X {
  int copy_count = 0;

  constexpr X() {}
  constexpr X(const X&) { copy_count++; }
};

constexpr void test_ref() {
  {
    X x{};
    std::optional<X&> o1(x);
    std::optional<X&> o2(o1);
    assert(o1.has_value() && o2.has_value());
    assert(x.copy_count == 0);
    assert(&*o1 == &*o2);
  }
}

void test_reference_extension() {
  using T = TestTypes::TestType;
  T::reset();
  {
    T t;
    T::reset_constructors();
    test_ref<T&>();
    test_ref<T&>(t);
    assert(T::alive == 1);
    assert(T::constructed == 0);
    assert(T::assigned == 0);
    assert(T::destroyed == 0);
  }
  assert(T::destroyed == 1);
  assert(T::alive == 0);
  {
    T t;
    const T& ct = t;
    T::reset_constructors();
    test_ref<T const&>();
    test_ref<T const&>(t);
    test_ref<T const&>(ct);
    assert(T::alive == 1);
    assert(T::constructed == 0);
    assert(T::assigned == 0);
    assert(T::destroyed == 0);
  }
  assert(T::alive == 0);
  assert(T::destroyed == 1);

#  if 0 // FIXME: optional<T&&> is not allowed.
  {
    static_assert(!std::is_copy_constructible<std::optional<T&&>>::value);
    static_assert(!std::is_copy_constructible<std::optional<T const&&>>::value);
  }
#  endif
}
#endif

constexpr bool test() {
  test<int>();
  test<int>(3);
  test<const int>(42);
  test<TrivialTestTypes::TestType>();
  test<TrivialTestTypes::TestType>(42);

  // FIXME: Why is this in ctor copy.pass.cpp?
  {
    constexpr std::optional<int> o1{4};
    constexpr std::optional<int> o2 = o1;
    static_assert(*o2 == 4, "");
  }

  {
    // LWG3836 https://wg21.link/LWG3836
    // std::optional<bool> conversion constructor optional(const optional<U>&)
    // should take precedence over optional(U&&) with operator bool
    {
      std::optional<bool> o1(false);
      std::optional<bool> o2(o1);
      assert(!o2.value());
    }
  }

#if TEST_STD_VER >= 26
  test_ref();

  // TODO: Enable once P3068R6 is implemented
#  if 0
  test_throwing_ctor();
#  endif
#endif

  return true;
}

void test_rt() {
  { // FIXME: Shouldn't this be able to pass in a constexpr context since C++17?
    using T = ConstexprTestTypes::TestType;
    test<T>();
    test<T>(42);
  }

  {
    using T = TestTypes::TestType;
    T::reset();
    const optional<T> rhs;
    assert(T::alive == 0);
    const optional<T> lhs(rhs);
    assert(lhs.has_value() == false);
    assert(T::alive == 0);
  }

  TestTypes::TestType::reset();

  {
    using T = TestTypes::TestType;
    T::reset();
    const optional<T> rhs(42);
    assert(T::alive == 1);
    assert(T::value_constructed == 1);
    assert(T::copy_constructed == 0);
    const optional<T> lhs(rhs);
    assert(lhs.has_value());
    assert(T::copy_constructed == 1);
    assert(T::alive == 2);
  }

  TestTypes::TestType::reset();
}

int main(int, char**) {
  test();
  static_assert(test());

  {
    test_rt();
  }

  {
    test_throwing_ctor();
  }

#if TEST_STD_VER >= 26
  {
    test_reference_extension();
  }
#endif

  return 0;
}
