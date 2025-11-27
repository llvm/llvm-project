//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <optional>

// constexpr optional(optional<T>&& rhs);

#include <cassert>
#include <optional>
#include <type_traits>

#include "test_macros.h"
#include "archetypes.h"

using std::optional;

template <class T, class... InitArgs>
constexpr bool test(InitArgs&&... args) {
  static_assert(std::is_trivially_move_constructible_v<T> == std::is_trivially_move_constructible_v<std::optional<T>>);
  const optional<T> orig(std::forward<InitArgs>(args)...);

  optional<T> lhs(orig);
  optional<T> rhs(std::move(lhs));

  assert(lhs.has_value() == rhs.has_value());
  assert(lhs.has_value() ? *rhs == *orig : true);

  return true;
}

TEST_CONSTEXPR_CXX26 void test_throwing_ctor() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Z {
    Z() : count(0) {}
    Z(Z&& o) : count(o.count + 1) {
      if (count == 2)
        throw 6;
    }
    int count;
  };
  Z z;
  optional<Z> rhs(std::move(z));
  try {
    optional<Z> lhs(std::move(rhs));
    assert(false);
  } catch (int i) {
    assert(i == 6);
  }
#endif
}

template <class T, class... InitArgs>
void test_ref(InitArgs&&... args) {
  optional<T> rhs(std::forward<InitArgs>(args)...);
  optional<T> lhs(std::move(rhs));

  assert(lhs.has_value() == rhs.has_value());
  assert(rhs.has_value() ? &*lhs == &*rhs : true);
}

struct F {
  int move_count = 0;

  constexpr F() {}
  constexpr F(F&&) { move_count++; }
};

#if TEST_STD_VER >= 26

constexpr void test_ref() {
  { // Test that moving from an optional<T&> doesn't also move the object it's referencing
    F f{};
    std::optional<F&> o1(f);
    std::optional<F&> o2(std::move(o1));
    assert(f.move_count == 0);
    assert(o1.has_value() && o2.has_value());
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
    T t;
    T::reset_constructors();
    test_ref<T&&>();
    test_ref<T&&>(std::move(t));
    assert(T::alive == 1);
    assert(T::constructed == 0);
    assert(T::assigned == 0);
    assert(T::destroyed == 0);
  }
  assert(T::alive == 0);
  assert(T::destroyed == 1);
  {
    T t;
    const T& ct = t;
    T::reset_constructors();
    test_ref<T const&&>();
    test_ref<T const&&>(std::move(t));
    test_ref<T const&&>(std::move(ct));
    assert(T::alive == 1);
    assert(T::constructed == 0);
    assert(T::assigned == 0);
    assert(T::destroyed == 0);
  }
  assert(T::alive == 0);
  assert(T::destroyed == 1);
  {
    static_assert(!std::is_copy_constructible_v<std::optional<T&&>>);
    static_assert(!std::is_copy_constructible_v<std::optional<T const&&>>);
  }
#  endif
}
#endif

constexpr bool test() {
  test<int>();
  test<int>(3);
  test<const int>(42);

  {
    using T = TrivialTestTypes::TestType;
    test<T>();
    test<T>(42);
  }

#if TEST_STD_VER >= 26
  test_ref();
#endif

#if TEST_STD_VER >= 26 && 0
  {
    test_throwing_ctor();
  }
#endif

  return true;
}

bool test_rt() {
  {
    using T = TestTypes::TestType;
    T::reset();
    optional<T> rhs;
    assert(T::alive == 0);
    const optional<T> lhs(std::move(rhs));
    assert(lhs.has_value() == false);
    assert(rhs.has_value() == false);
    assert(T::alive == 0);
  }

  TestTypes::TestType::reset();

  {
    using T = TestTypes::TestType;
    T::reset();
    optional<T> rhs(42);
    assert(T::alive == 1);
    assert(T::value_constructed == 1);
    assert(T::move_constructed == 0);
    const optional<T> lhs(std::move(rhs));
    assert(lhs.has_value());
    assert(rhs.has_value());
    assert(lhs.value().value == 42);
    assert(rhs.value().value == -1);
    assert(T::move_constructed == 1);
    assert(T::alive == 2);
  }

  TestTypes::TestType::reset();

  { // TODO: Why doesn't this pass in a C++17 constexpr context?
    using namespace ConstexprTestTypes;
    test<TestType>();
    test<TestType>(42);
  }

  {
    struct ThrowsMove {
      ThrowsMove() noexcept(false) {}
      ThrowsMove(ThrowsMove const&) noexcept(false) {}
      ThrowsMove(ThrowsMove&&) noexcept(false) {}
    };
    static_assert(!std::is_nothrow_move_constructible<optional<ThrowsMove>>::value, "");
    struct NoThrowMove {
      NoThrowMove() noexcept(false) {}
      NoThrowMove(NoThrowMove const&) noexcept(false) {}
      NoThrowMove(NoThrowMove&&) noexcept(true) {}
    };
    static_assert(std::is_nothrow_move_constructible<optional<NoThrowMove>>::value, "");
  }

  {
    test_throwing_ctor();
  }

#if TEST_STD_VER >= 26
  {
    test_reference_extension();
  }
#endif

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());

  {
    test_rt();
  }

  return 0;
}
