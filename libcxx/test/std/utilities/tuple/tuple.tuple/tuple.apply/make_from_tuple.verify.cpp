//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <tuple>

// template <class T, class Tuple> constexpr T make_from_tuple(Tuple&&);
// Mandates: If tuple_size_v<remove_reference_t<Tuple>> is 1, then reference_constructs_from_temporary_v<T, decltype(get<0>(declval<Tuple>()))> is false.

#include <tuple>
#include <utility>

#include "test_macros.h"

void test() {
  // FreeBSD ci use clang 19.1.1, which hasn't implement __reference_constructs_from_temporary.
  // The static_assert inner std::make_from_tuple will not triggered.
#if __has_builtin(__reference_constructs_from_temporary)
  // expected-error@*:* {{static assertion failed}}
#endif

  // Turns to an error since C++26 (Disallow Binding a Returned Glvalue to a Temporary https://wg21.link/P2748R5).
#if TEST_STD_VER >= 26
  // expected-error@tuple:* {{returning reference to local temporary object}}
#else
  // expected-warning@tuple:* {{returning reference to local temporary object}}
#endif
  std::ignore = std::make_from_tuple<const int&>(std::tuple<char>{});
}
