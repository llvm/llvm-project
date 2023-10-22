//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//
// constexpr expected();

// Constraints: is_default_constructible_v<T> is true.
//
// Effects: Value-initializes val.
// Postconditions: has_value() is true.
//
// Throws: Any exception thrown by the initialization of val.

#include <cassert>
#include <expected>
#include <type_traits>

#include "test_macros.h"
#include "../../types.h"

struct NoDedefaultCtor {
  NoDedefaultCtor() = delete;
};

// Test constraints
static_assert(std::is_default_constructible_v<std::expected<int, int>>);
static_assert(!std::is_default_constructible_v<std::expected<NoDedefaultCtor, int>>);

struct MyInt {
  int i;
  friend constexpr bool operator==(const MyInt&, const MyInt&) = default;
};

template <class T, class E>
constexpr void testDefaultCtor() {
  std::expected<T, E> e;
  assert(e.has_value());
  assert(e.value() == T());
}

template <class T>
constexpr void testTypes() {
  testDefaultCtor<T, bool>();
  testDefaultCtor<T, int>();
  testDefaultCtor<T, NoDedefaultCtor>();
}

constexpr bool test() {
  testTypes<int>();
  testTypes<MyInt>();
  testTypes<TailClobberer<0>>();
  return true;
}

void testException() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  struct Throwing {
    Throwing() { throw Except{}; };
  };

  try {
    std::expected<Throwing, int> u;
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
