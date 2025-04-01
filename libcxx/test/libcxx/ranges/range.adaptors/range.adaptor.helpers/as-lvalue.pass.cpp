//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class T>
// constexpr T& as-lvalue(T&& t) { // exposition only

#include <__utility/as_lvalue.h>
#include <type_traits>
#include <utility>

constexpr bool test() {
  // Check glvalue
  {
    int lvalue{};
    [[maybe_unused]] decltype(auto) check = std::__as_lvalue(lvalue);
    static_assert(std::is_same<decltype(check), int&>::value, "");
  }

  // Check xvalue
  {
    int xvalue{};
    [[maybe_unused]] decltype(auto) check = std::__as_lvalue(std::move(xvalue));
    static_assert(std::is_same<decltype(check), int&>::value, "");
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
