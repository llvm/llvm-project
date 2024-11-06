//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class E2>
// friend constexpr bool operator==(const unexpected& x, const unexpected<E2>& y);
//
// Mandates: The expression x.error() == y.error() is well-formed and its result is convertible to bool.
//
// Returns: x.error() == y.error().

#include <cassert>
#include <concepts>
#include <expected>
#include <utility>

struct Error{
  int i;
  friend constexpr bool operator==(const Error&, const Error&) = default;
};

constexpr bool test() {
  std::unexpected<Error> unex1(Error{2});
  std::unexpected<Error> unex2(Error{3});
  std::unexpected<Error> unex3(Error{2});
  assert(unex1 == unex3);
  assert(unex1 != unex2);
  assert(unex2 != unex3);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
