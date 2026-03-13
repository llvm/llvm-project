//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class U, class... Args>
//   constexpr explicit unexpected(in_place_t, initializer_list<U> il, Args&&... args);
//
// Constraints: is_constructible_v<E, initializer_list<U>&, Args...> is true.
//
// Effects: Direct-non-list-initializes unex with il, std::forward<Args>(args)....
//
// Throws: Any exception thrown by the initialization of unex.

#include <algorithm>
#include <cassert>
#include <concepts>
#include <expected>
#include <utility>

#include "test_macros.h"

struct Arg {
  int i;
  constexpr Arg(int ii) : i(ii) {}
  constexpr Arg(const Arg& other) : i(other.i) {}
  constexpr Arg(Arg&& other) : i(other.i) { other.i = 0; }
};

struct Error {
  std::initializer_list<int> list;
  Arg arg;
  constexpr explicit Error(std::initializer_list<int> l, const Arg& a) : list(l), arg(a) {}
  constexpr explicit Error(std::initializer_list<int> l, Arg&& a) : list(l), arg(std::move(a)) {}
};

// Test Constraints:
static_assert(std::constructible_from<std::unexpected<Error>, std::in_place_t, std::initializer_list<int>, Arg>);

// !is_constructible_v<E, initializer_list<U>&, Args...>
struct Foo {};
static_assert(!std::constructible_from<std::unexpected<Error>, std::in_place_t, std::initializer_list<double>, Arg>);

// test explicit
template <class T>
void conversion_test(T);

template <class T, class... Args>
concept ImplicitlyConstructible = requires(Args&&... args) { conversion_test<T>({std::forward<Args>(args)...}); };

static_assert(ImplicitlyConstructible<int, int>);
static_assert(!ImplicitlyConstructible<std::unexpected<Error>, std::in_place_t, std::initializer_list<int>, Arg>);

constexpr bool test() {
  // lvalue
  {
    Arg a{5};
    auto l = {1, 2, 3};
    std::unexpected<Error> unex(std::in_place, l, a);
    assert(unex.error().arg.i == 5);
    assert(std::ranges::equal(unex.error().list, l));
    assert(a.i == 5);
  }

  // rvalue
  {
    Arg a{5};
    auto l = {1, 2, 3};
    std::unexpected<Error> unex(std::in_place, l, std::move(a));
    assert(unex.error().arg.i == 5);
    assert(std::ranges::equal(unex.error().list, l));
    assert(a.i == 0);
  }

  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Except {};

  struct Throwing {
    Throwing(std::initializer_list<int>, int) { throw Except{}; }
  };

  try {
    std::unexpected<Throwing> u(std::in_place, {1, 2}, 5);
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
