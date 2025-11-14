//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <utility>

// template<size_t I, class T, T... Values>
//   [[nodiscard]] constexpr T get(integer_sequence<T, Values...>) noexcept;

// check that get is marked [[nodiscard]]

#include <utility>

void f() {
  std::index_sequence<1> seq;
  get<0>(seq);    // expected-warning {{ignoring return value of function}}
}
