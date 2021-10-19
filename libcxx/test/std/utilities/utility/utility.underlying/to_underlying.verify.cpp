//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// [utility.underlying], to_underlying
// template <class T>
//     constexpr underlying_type_t<T> to_underlying( T value ) noexcept; // C++2b

#include <utility>

struct S {};

int main(int, char**) {
  std::to_underlying(125); // expected-error {{no matching function for call}}
  std::to_underlying(S{}); // expected-error {{no matching function for call}}

  return 0;
}
