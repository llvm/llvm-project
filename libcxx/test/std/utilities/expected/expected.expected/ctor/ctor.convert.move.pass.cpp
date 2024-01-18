//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  template<class U, class G>
//    constexpr explicit(see below) expected(expected<U, G>&&);
//
// Let:
// - UF be const U
// - GF be const G
//
// Constraints:
// - is_constructible_v<T, UF> is true; and
// - is_constructible_v<E, GF> is true; and
// - is_constructible_v<T, expected<U, G>&> is false; and
// - is_constructible_v<T, expected<U, G>> is false; and
// - is_constructible_v<T, const expected<U, G>&> is false; and
// - is_constructible_v<T, const expected<U, G>> is false; and
// - is_convertible_v<expected<U, G>&, T> is false; and
// - is_convertible_v<expected<U, G>&&, T> is false; and
// - is_convertible_v<const expected<U, G>&, T> is false; and
// - is_convertible_v<const expected<U, G>&&, T> is false; and
// - is_constructible_v<unexpected<E>, expected<U, G>&> is false; and
// - is_constructible_v<unexpected<E>, expected<U, G>> is false; and
// - is_constructible_v<unexpected<E>, const expected<U, G>&> is false; and
// - is_constructible_v<unexpected<E>, const expected<U, G>> is false.
//
// Effects: If rhs.has_value(), direct-non-list-initializes val with std::forward<UF>(*rhs). Otherwise, direct-non-list-initializes unex with std::forward<GF>(rhs.error()).
//
// Postconditions: rhs.has_value() is unchanged; rhs.has_value() == this->has_value() is true.
//
// Throws: Any exception thrown by the initialization of val or unex.
//
// Remarks: The expression inside explicit is equivalent to !is_convertible_v<UF, T> || !is_convertible_v<GF, E>.

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "MoveOnly.h"
#include "test_macros.h"
#include "../../types.h"

// Test Constraints:
template <class T1, class Err1, class T2, class Err2>
concept canCstrFromExpected = std::is_constructible_v<std::expected<T1, Err1>, std::expected<T2, Err2>&&>;

struct CtorFromInt {
  CtorFromInt(int);
};

static_assert(canCstrFromExpected<CtorFromInt, int, int, int>);

struct NoCtorFromInt {};

// !is_constructible_v<T, UF>
static_assert(!canCstrFromExpected<NoCtorFromInt, int, int, int>);

// !is_constructible_v<E, GF>
static_assert(!canCstrFromExpected<int, NoCtorFromInt, int, int>);

template <class T>
struct CtorFrom {
  explicit CtorFrom(int)
    requires(!std::same_as<T, int>);
  explicit CtorFrom(T);
  explicit CtorFrom(auto&&) = delete;
};

// is_constructible_v<T, expected<U, G>&>
static_assert(!canCstrFromExpected<CtorFrom<std::expected<int, int>&>, int, int, int>);

// is_constructible_v<T, expected<U, G>>
// note that this is true because it is covered by the other overload
//   template<class U = T> constexpr explicit(see below) expected(U&& v);
// The fact that it is not ambiguous proves that the overload under testing is removed
static_assert(canCstrFromExpected<CtorFrom<std::expected<int, int>&&>, int, int, int>);

// is_constructible_v<T, expected<U, G>&>
static_assert(!canCstrFromExpected<CtorFrom<std::expected<int, int> const&>, int, int, int>);

// is_constructible_v<T, expected<U, G>>
static_assert(!canCstrFromExpected<CtorFrom<std::expected<int, int> const&&>, int, int, int>);

template <class T>
struct ConvertFrom {
  ConvertFrom(int)
    requires(!std::same_as<T, int>);
  ConvertFrom(T);
  ConvertFrom(auto&&) = delete;
};

// is_convertible_v<expected<U, G>&, T>
static_assert(!canCstrFromExpected<ConvertFrom<std::expected<int, int>&>, int, int, int>);

// is_convertible_v<expected<U, G>&&, T>
// note that this is true because it is covered by the other overload
//   template<class U = T> constexpr explicit(see below) expected(U&& v);
// The fact that it is not ambiguous proves that the overload under testing is removed
static_assert(canCstrFromExpected<ConvertFrom<std::expected<int, int>&&>, int, int, int>);

// is_convertible_v<const expected<U, G>&, T>
static_assert(!canCstrFromExpected<ConvertFrom<std::expected<int, int> const&>, int, int, int>);

// is_convertible_v<const expected<U, G>&&, T>
static_assert(!canCstrFromExpected<ConvertFrom<std::expected<int, int> const&&>, int, int, int>);

// is_constructible_v<unexpected<E>, expected<U, G>&>
static_assert(!canCstrFromExpected<int, CtorFrom<std::expected<int, int>&>, int, int>);

// is_constructible_v<unexpected<E>, expected<U, G>>
static_assert(!canCstrFromExpected<int, CtorFrom<std::expected<int, int>&&>, int, int>);

// is_constructible_v<unexpected<E>, const expected<U, G>&> is false
static_assert(!canCstrFromExpected<int, CtorFrom<std::expected<int, int> const&>, int, int>);

// is_constructible_v<unexpected<E>, const expected<U, G>>
static_assert(!canCstrFromExpected<int, CtorFrom<std::expected<int, int> const&&>, int, int>);

// test explicit
static_assert(std::is_convertible_v<std::expected<int, int>&&, std::expected<short, long>>);

// !is_convertible_v<UF, T>
static_assert(std::is_constructible_v<std::expected<CtorFrom<int>, int>, std::expected<int, int>&&>);
static_assert(!std::is_convertible_v<std::expected<int, int>&&, std::expected<CtorFrom<int>, int>>);

// !is_convertible_v<GF, E>.
static_assert(std::is_constructible_v<std::expected<int, CtorFrom<int>>, std::expected<int, int>&&>);
static_assert(!std::is_convertible_v<std::expected<int, int>&&, std::expected<int, CtorFrom<int>>>);

struct Data {
  MoveOnly data;
  constexpr Data(MoveOnly&& m) : data(std::move(m)) {}
};

constexpr bool test() {
  // convert the value
  {
    std::expected<MoveOnly, int> e1(5);
    std::expected<Data, int> e2 = std::move(e1);
    assert(e2.has_value());
    assert(e2.value().data.get() == 5);
    assert(e1.has_value());
    assert(e1.value().get() == 0);
  }

  // convert the error
  {
    std::expected<int, MoveOnly> e1(std::unexpect, 5);
    std::expected<int, Data> e2 = std::move(e1);
    assert(!e2.has_value());
    assert(e2.error().data.get() == 5);
    assert(!e1.has_value());
    assert(e1.error().get() == 0);
  }

  // convert TailClobberer
  {
    std::expected<TailClobbererNonTrivialMove<0>, char> e1;
    std::expected<TailClobberer<0>, char> e2 = std::move(e1);
    assert(e2.has_value());
    assert(e1.has_value());
  }

  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct ThrowingInt {
    ThrowingInt(int) { throw Except{}; }
  };

  // throw on converting value
  {
    const std::expected<int, int> e1;
    try {
      [[maybe_unused]] std::expected<ThrowingInt, int> e2 = e1;
      assert(false);
    } catch (Except) {
    }
  }

  // throw on converting error
  {
    const std::expected<int, int> e1(std::unexpect);
    try {
      [[maybe_unused]] std::expected<int, ThrowingInt> e2 = e1;
      assert(false);
    } catch (Except) {
    }
  }

#endif // TEST_HAS_NO_EXCEPTIONS
}

int main(int, char**) {
  test();
  static_assert(test());
  testException();
  return 0;
}
