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

#include <concepts>
#include <utility>

constexpr bool test() {
  // Check glvalue
  {
    int lvalue{};
    [[maybe_unused]] std::same_as<int&> decltype(auto) check = std::__as_lvalue(lvalue);
  }

  // Check xvalue
  {
    int xvalue{};
    [[maybe_unused]] std::same_as<int&> decltype(auto) check = std::__as_lvalue(std::move(xvalue));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
