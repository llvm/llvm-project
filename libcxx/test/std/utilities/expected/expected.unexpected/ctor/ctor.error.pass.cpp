//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class Err = E>
//   constexpr explicit unexpected(Err&& e);
//
// Constraints:
// - is_same_v<remove_cvref_t<Err>, unexpected> is false; and
// - is_same_v<remove_cvref_t<Err>, in_place_t> is false; and
// - is_constructible_v<E, Err> is true.
//
// Effects: Direct-non-list-initializes unex with std::forward<Err>(e).
// Throws: Any exception thrown by the initialization of unex.

#include <cassert>
#include <concepts>
#include <expected>
#include <utility>

#include "test_macros.h"

// Test Constraints:
static_assert(std::constructible_from<std::unexpected<int>, int>);

// is_same_v<remove_cvref_t<Err>, unexpected>
struct CstrFromUnexpected {
  CstrFromUnexpected(CstrFromUnexpected const&) = delete;
  CstrFromUnexpected(std::unexpected<CstrFromUnexpected> const&);
};
static_assert(!std::constructible_from<std::unexpected<CstrFromUnexpected>, std::unexpected<CstrFromUnexpected>>);

// is_same_v<remove_cvref_t<Err>, in_place_t>
struct CstrFromInplace {
  CstrFromInplace(std::in_place_t);
};
static_assert(!std::constructible_from<std::unexpected<CstrFromInplace>, std::in_place_t>);

// !is_constructible_v<E, Err>
struct Foo {};
static_assert(!std::constructible_from<std::unexpected<Foo>, int>);

// test explicit
static_assert(std::convertible_to<int, int>);
static_assert(!std::convertible_to<int, std::unexpected<int>>);

struct Error {
  int i;
  constexpr Error(int ii) : i(ii) {}
  constexpr Error(const Error& other) : i(other.i) {}
  constexpr Error(Error&& other) : i(other.i) { other.i = 0; }
  Error(std::initializer_list<Error>) { assert(false); }
};

constexpr bool test() {
  // lvalue
  {
    Error e(5);
    std::unexpected<Error> unex(e);
    assert(unex.error().i == 5);
    assert(e.i == 5);
  }

  // rvalue
  {
    Error e(5);
    std::unexpected<Error> unex(std::move(e));
    assert(unex.error().i == 5);
    assert(e.i == 0);
  }

  // Direct-non-list-initializes: does not trigger initializer_list overload
  {
    Error e(5);
    [[maybe_unused]] std::unexpected<Error> unex(e);
  }

  // Test default template argument.
  // Without it, the template parameter cannot be deduced from an initializer list
  {
    struct Bar {
      int i;
      int j;
      constexpr Bar(int ii, int jj) : i(ii), j(jj) {}
    };
    std::unexpected<Bar> ue({5, 6});
    assert(ue.error().i == 5);
    assert(ue.error().j == 6);
  }

  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Except {};

  struct Throwing {
    Throwing() = default;
    Throwing(const Throwing&) { throw Except{}; }
  };

  Throwing t;
  try {
    std::unexpected<Throwing> u(t);
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
