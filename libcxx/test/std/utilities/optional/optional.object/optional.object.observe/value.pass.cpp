//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <optional>

// constexpr T& optional<T>::value() &;
// constexpr const T& optional<T>::value() const&;
// constexpr T&& optional<T>::value() &&;
// constexpr const T&& optional<T>::value() const T&&;
// constexpr T& optional<T&>::value() const;

#include <cassert>
#include <optional>
#include <utility>

#include "test_macros.h"

struct X {
  constexpr X() = default;
  constexpr int test() const& { return 3; }
  constexpr int test() & { return 4; }
  constexpr int test() const&& { return 5; }
  constexpr int test() && { return 6; }
};

template <typename T>
constexpr void test_contract() {
  std::optional<T> opt;

  ASSERT_SAME_TYPE(decltype(opt.value()), T&);
  ASSERT_SAME_TYPE(decltype(std::as_const(opt).value()), const T&);
  ASSERT_SAME_TYPE(decltype(std::move(opt).value()), T&&);
  ASSERT_SAME_TYPE(decltype(std::move(std::as_const(opt)).value()), const T&&);

  ASSERT_NOT_NOEXCEPT(opt.value());
  ASSERT_NOT_NOEXCEPT(std::as_const(opt).value());
  ASSERT_NOT_NOEXCEPT(std::move(opt).value());
  ASSERT_NOT_NOEXCEPT(std::move(std::as_const(opt)).value());
}

#if TEST_STD_VER >= 26
template <typename T>
constexpr void test_ref_contract() {
  std::optional<T&> opt;

  ASSERT_SAME_TYPE(decltype(opt.value()), T&);
  ASSERT_SAME_TYPE(decltype(std::as_const(opt).value()), T&);
  ASSERT_SAME_TYPE(decltype(std::move(opt).value()), T&);
  ASSERT_SAME_TYPE(decltype(std::move(std::as_const(opt)).value()), T&);

  ASSERT_NOT_NOEXCEPT(opt.value());
  ASSERT_NOT_NOEXCEPT(std::as_const(opt).value());
  ASSERT_NOT_NOEXCEPT(std::move(opt).value());
  ASSERT_NOT_NOEXCEPT(std::move(std::as_const(opt)).value());
}
#endif

constexpr bool test() {
  test_contract<int>();
  test_contract<float>();
  test_contract<double>();
  test_contract<X>();

  test_contract<const int>();
  test_contract<const float>();
  test_contract<const double>();
  test_contract<const X>();

  {
    std::optional<X> opt(X{});

    assert(std::as_const(opt).value().test() == 3);
    assert(opt.value().test() == 4);
    assert(std::move(std::as_const(opt)).value().test() == 5);
    assert(std::move(opt).value().test() == 6);
  }

#if TEST_STD_VER >= 26
  test_ref_contract<int>();
  test_ref_contract<float>();
  test_ref_contract<double>();
  test_ref_contract<X>();

  test_ref_contract<const int>();
  test_ref_contract<const float>();
  test_ref_contract<const double>();
  test_ref_contract<const X>();

  {
    X x;
    std::optional<X&> o1{x};
    assert(o1.value().test() == 4);
    assert(std::as_const(o1).value().test() == 4);
    assert(std::move(o1).value().test() == 4);
    assert(std::move(std::as_const(o1)).value().test() == 4);
  }
  {
    X x;
    std::optional<const X&> o2{x};
    assert(o2.value().test() == 3);
    assert(std::as_const(o2).value().test() == 3);
    assert(std::move(o2).value().test() == 3);
    assert(std::move(std::as_const(o2)).value().test() == 3);
  }
#endif

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    std::optional<X> opt;
    try {
      (void)opt.value();
      assert(false);
    } catch (const std::bad_optional_access&) {
    }

    try {
      (void)std::as_const(opt).value();
      assert(false);
    } catch (const std::bad_optional_access&) {
    }

    try {
      (void)std::move(opt).value();
      assert(false);
    } catch (const std::bad_optional_access&) {
    }

    try {
      (void)std::move(std::as_const(opt)).value();
      assert(false);
    } catch (const std::bad_optional_access&) {
    }
  }

  {
    std::optional<X> opt(X{});
    try {
      (void)opt.value();
    } catch (const std::bad_optional_access&) {
      assert(false);
    }

    try {
      (void)std::as_const(opt).value();
    } catch (const std::bad_optional_access&) {
      assert(false);
    }

    try {
      (void)std::move(opt).value();
    } catch (const std::bad_optional_access&) {
      assert(false);
    }

    try {
      (void)std::move(std::as_const(opt)).value();
    } catch (const std::bad_optional_access&) {
      assert(false);
    }
  }
#endif

  return 0;
}
