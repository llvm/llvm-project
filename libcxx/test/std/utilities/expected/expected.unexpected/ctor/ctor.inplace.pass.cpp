//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class... Args>
//   constexpr explicit unexpected(in_place_t, Args&&... args);
//
// Constraints: is_constructible_v<E, Args...> is true.
//
// Effects: Direct-non-list-initializes unex with std::forward<Args>(args)....
//
// Throws: Any exception thrown by the initialization of unex.

#include <cassert>
#include <concepts>
#include <expected>
#include <utility>

#include "test_macros.h"

// Test Constraints:
static_assert(std::constructible_from<std::unexpected<int>, std::in_place_t, int>);

// !is_constructible_v<E, Args...>
struct Foo {};
static_assert(!std::constructible_from<std::unexpected<Foo>, std::in_place_t, int>);

// test explicit
template <class T>
void conversion_test(T);

template <class T, class... Args>
concept ImplicitlyConstructible = requires(Args&&... args) { conversion_test<T>({std::forward<Args>(args)...}); };

static_assert(ImplicitlyConstructible<int, int>);
static_assert(!ImplicitlyConstructible<std::unexpected<int>, std::in_place_t, int>);

struct Arg {
  int i;
  constexpr Arg(int ii) : i(ii) {}
  constexpr Arg(const Arg& other) : i(other.i) {}
  constexpr Arg(Arg&& other) : i(other.i) { other.i = 0; }
};

struct Error {
  Arg arg;
  constexpr explicit Error(const Arg& a) : arg(a) {}
  constexpr explicit Error(Arg&& a) : arg(std::move(a)) {}
  Error(std::initializer_list<Error>) :arg(0){ assert(false); }
};

constexpr bool test() {
  // lvalue
  {
    Arg a{5};
    std::unexpected<Error> unex(std::in_place, a);
    assert(unex.error().arg.i == 5);
    assert(a.i == 5);
  }

  // rvalue
  {
    Arg a{5};
    std::unexpected<Error> unex(std::in_place, std::move(a));
    assert(unex.error().arg.i == 5);
    assert(a.i == 0);
  }

  // Direct-non-list-initializes: does not trigger initializer_list overload
  {
    Error e(5);
    [[maybe_unused]] std::unexpected<Error> unex(std::in_place, e);
  }
  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Except {};

  struct Throwing {
    Throwing(int) { throw Except{}; }
  };

  try {
    std::unexpected<Throwing> u(std::in_place, 5);
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
