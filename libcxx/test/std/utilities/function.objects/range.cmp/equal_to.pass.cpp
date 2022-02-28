//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// <functional>

// ranges::equal_to

#include <functional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "compare_types.h"
#include "MoveOnly.h"
#include "pointer_comparison_test_helper.h"

struct NotEqualityComparable {
  friend bool operator==(const NotEqualityComparable&, const NotEqualityComparable&);
  friend bool operator!=(const NotEqualityComparable&, const NotEqualityComparable&) = delete;
};

static_assert(!std::is_invocable_v<std::ranges::equal_to, NotEqualityComparable, NotEqualityComparable>);
static_assert(!std::is_invocable_v<std::ranges::equal_to, int, MoveOnly>);
static_assert(std::is_invocable_v<std::ranges::equal_to, explicit_operators, explicit_operators>);

static_assert(requires { typename std::ranges::equal_to::is_transparent; });

constexpr bool test() {
  auto fn = std::ranges::equal_to();

  assert(fn(MoveOnly(42), MoveOnly(42)));

  ForwardingTestObject a;
  ForwardingTestObject b;
  assert(!fn(a, b));
  assert(fn(std::move(a), std::move(b)));

  assert(!fn(1, 2));
  assert(!fn(2, 1));
  assert(fn(2, 2));

  assert(!fn(2, 1L));

  return true;
}

int main(int, char**) {

  test();
  static_assert(test());

  // test total ordering of int* for equal_to<int*> and equal_to<void>.
  do_pointer_comparison_test(std::ranges::equal_to());

  return 0;
}
