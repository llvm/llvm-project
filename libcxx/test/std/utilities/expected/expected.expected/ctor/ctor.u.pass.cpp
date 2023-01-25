//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class U = T>
//   constexpr explicit(!is_convertible_v<U, T>) expected(U&& v);
//
// Constraints:
// - is_same_v<remove_cvref_t<U>, in_place_t> is false; and
// - is_same_v<expected, remove_cvref_t<U>> is false; and
// - remove_cvref_t<U> is not a specialization of unexpected; and
// - is_constructible_v<T, U> is true.
//
// Effects: Direct-non-list-initializes val with std::forward<U>(v).
//
// Postconditions: has_value() is true.
//
// Throws: Any exception thrown by the initialization of val.

#include <cassert>
#include <expected>
#include <type_traits>
#include <utility>

#include "MoveOnly.h"
#include "test_macros.h"

// Test Constraints:
static_assert(std::is_constructible_v<std::expected<int, int>, int>);

// is_same_v<remove_cvref_t<U>, in_place_t>
struct FromJustInplace {
  FromJustInplace(std::in_place_t);
};
static_assert(!std::is_constructible_v<std::expected<FromJustInplace, int>, std::in_place_t>);
static_assert(!std::is_constructible_v<std::expected<FromJustInplace, int>, std::in_place_t const&>);

// is_same_v<expected, remove_cvref_t<U>>
// Note that result is true because it is covered by the constructors that take expected
static_assert(std::is_constructible_v<std::expected<int, int>, std::expected<int, int>&>);

// remove_cvref_t<U> is a specialization of unexpected
// Note that result is true because it is covered by the constructors that take unexpected
static_assert(std::is_constructible_v<std::expected<int, int>, std::unexpected<int>&>);

// !is_constructible_v<T, U>
struct foo {};
static_assert(!std::is_constructible_v<std::expected<int, int>, foo>);

// test explicit(!is_convertible_v<U, T>)
struct NotConvertible {
  explicit NotConvertible(int);
};
static_assert(std::is_convertible_v<int, std::expected<int, int>>);
static_assert(!std::is_convertible_v<int, std::expected<NotConvertible, int>>);

struct CopyOnly {
  int i;
  constexpr CopyOnly(int ii) : i(ii) {}
  CopyOnly(const CopyOnly&) = default;
  CopyOnly(CopyOnly&&)      = delete;
  friend constexpr bool operator==(const CopyOnly& mi, int ii) { return mi.i == ii; }
};

template <class T>
constexpr void testInt() {
  std::expected<T, int> e(5);
  assert(e.has_value());
  assert(e.value() == 5);
}

template <class T>
constexpr void testLValue() {
  T t(5);
  std::expected<T, int> e(t);
  assert(e.has_value());
  assert(e.value() == 5);
}

template <class T>
constexpr void testRValue() {
  std::expected<T, int> e(T(5));
  assert(e.has_value());
  assert(e.value() == 5);
}

constexpr bool test() {
  testInt<int>();
  testInt<CopyOnly>();
  testInt<MoveOnly>();
  testLValue<int>();
  testLValue<CopyOnly>();
  testRValue<int>();
  testRValue<MoveOnly>();

  // Test default template argument.
  // Without it, the template parameter cannot be deduced from an initializer list
  {
    struct Bar {
      int i;
      int j;
      constexpr Bar(int ii, int jj) : i(ii), j(jj) {}
    };

    std::expected<Bar, int> e({5, 6});
    assert(e.value().i == 5);
    assert(e.value().j == 6);
  }

  // this is a confusing example, but the behaviour
  // is exactly what is specified in the spec
  // see https://cplusplus.github.io/LWG/issue3836
  {
    struct BaseError {};
    struct DerivedError : BaseError {};

    std::expected<bool, DerivedError> e1(false);
    std::expected<bool, BaseError> e2(e1);
    assert(e2.has_value());
    assert(e2.value()); // yes, e2 holds "true"
  }
  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Except {};

  struct Throwing {
    Throwing(int) { throw Except{}; };
  };

  try {
    std::expected<Throwing, int> u(5);
    assert(false);
  } catch (Except) {
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test();
  static_assert(test());
  testException();
  return 0;
}
