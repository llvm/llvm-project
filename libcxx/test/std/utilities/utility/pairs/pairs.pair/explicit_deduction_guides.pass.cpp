//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <utility>

// template<class T1, class T2>
//   pair(T1, T2) -> pair<T1, T2>;

// Test that the explicit deduction guide for std::pair correctly decays function lvalues and
// behaves different from std::make_pair.

#include <cassert>
#include <functional>
#include <type_traits>
#include <utility>

#include "test_macros.h"

void dummy() {}

constexpr void test_decay() {
  char arr[1]{};
  std::pair pr(arr, dummy);

  ASSERT_SAME_TYPE(decltype(pr), std::pair<char*, void (*)()>);

  assert(pr == std::make_pair(arr, dummy));
}

TEST_CONSTEXPR_CXX20 void test_unwrap() {
  int n = 0;
  std::pair pr(std::ref(n), dummy);

  ASSERT_SAME_TYPE(decltype(pr), std::pair<std::reference_wrapper<int>, void (*)()>);
  static_assert(!std::is_same_v<decltype(pr), decltype(std::make_pair(std::ref(n), dummy))>);

  assert(&(pr.first.get()) == &n);
  assert(pr.second == dummy);
}

constexpr bool test() {
  test_decay();
  if (TEST_STD_AT_LEAST_20_OR_RUNTIME_EVALUATED)
    test_unwrap();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
