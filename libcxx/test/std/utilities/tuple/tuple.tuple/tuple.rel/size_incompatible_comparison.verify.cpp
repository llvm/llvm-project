//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Disabled in C++26 and later because tuple comparison between different sizes is constrained since P2944R3.
// UNSUPPORTED: std-at-least-cxx26

// <tuple>

// template <class... Types> class tuple;

// template<class... TTypes, class... UTypes>
//   bool
//   operator==(const tuple<TTypes...>& t, const tuple<UTypes...>& u);
// template<class... TTypes, class... UTypes>
//   bool
//   operator<(const tuple<TTypes...>& t, const tuple<UTypes...>& u);

// UNSUPPORTED: c++03

#include <tuple>

void f(std::tuple<int> t1, std::tuple<int, long> t2) {
  // We test only the core comparison operators and trust that the others
  // fall back on the same implementations prior to C++20.
  static_cast<void>(t1 == t2); // expected-error@*:* {{}}
  static_cast<void>(t1 < t2); // expected-error@*:* {{}}
}
