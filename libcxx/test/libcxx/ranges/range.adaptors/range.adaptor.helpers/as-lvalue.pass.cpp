//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// template<class T>
// constexpr T& as-lvalue(T&& t) { // exposition only

#include <cassert>
#include <type_traits>
#include <utility>

constexpr bool test(int value = 0) {
  static_assert(std::is_same<decltype(std::__as_lvalue(value)), int&>::value, "");
  static_assert(std::is_same<decltype(std::__as_lvalue(std::move(value))), int&>::value, "");

  return (assert(&std::__as_lvalue(value) == &value), assert(&std::__as_lvalue(std::move(value)) == &value), true);
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
