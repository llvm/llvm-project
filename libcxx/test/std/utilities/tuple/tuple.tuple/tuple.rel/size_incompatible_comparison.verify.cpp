//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template<class... TTypes, class... UTypes>
// bool operator==(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
//
// template<class... TTypes, class... UTypes>
// bool operator<(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
//
// template<tuple-like UTuple>
// friend constexpr bool operator==(const tuple& t, const UTuple& u); // since C++23

// <tuple> is not supported in C++03. In C++26 and later, tuple comparison between
// different sizes is constrained (since P2944R3), so this test does not apply.
// REQUIRES: c++11 || c++14 || c++17 || c++20 || c++23

#include <array>
#include <tuple>

#include "test_macros.h"

void f(std::tuple<int> t1, std::tuple<int, long> t2) {
  // We test only the core comparison operators and trust that the others
  // fall back on the same implementations prior to C++20.
  static_cast<void>(t1 == t2); // expected-error@*:* {{}}
  static_cast<void>(t1 < t2);  // expected-error@*:* {{}}

#if TEST_STD_VER >= 23
  std::array<int, 2> a{};     // a tuple-like with a different size
  static_cast<void>(t1 == a); // expected-error@*:* {{}}
#endif
}
