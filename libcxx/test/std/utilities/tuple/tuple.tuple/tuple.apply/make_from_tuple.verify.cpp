//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <tuple>

// template <class T, class Tuple> constexpr T make_from_tuple(Tuple&&);
// Mandates: If tuple_size_v<remove_reference_t<Tuple>> is 1, then reference_constructs_from_temporary_v<T, decltype(get<0>(declval<Tuple>()))> is false.

#include <tuple>
#include <utility>

void test() {
  // expected-error@*:* {{static assertion failed}}
  // expected-error@*:* {{returning reference to local temporary object}}
  std::ignore = std::make_from_tuple<const int&>(std::tuple<char>{});
}
