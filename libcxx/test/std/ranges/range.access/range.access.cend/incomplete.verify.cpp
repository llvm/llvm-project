//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges
// REQUIRES: libc++

// Test the libc++ specific behavior that we provide a better diagnostic when calling
// std::ranges::cend on an array of incomplete type.

#include <ranges>

#include <type_traits>

using cend_t = decltype(std::ranges::cend);

template <class T> void f() requires std::invocable<cend_t&, T> { }
template <class T> void f() { }

void test() {
  struct incomplete;
  f<incomplete(&)[]>();
  // expected-error@*:* {{"`std::ranges::begin` is SFINAE-unfriendly on arrays of an incomplete type."}}
  // expected-error@*:* {{"`std::ranges::end` is SFINAE-unfriendly on arrays of an incomplete type."}}
  f<incomplete(&)[10]>();
  // expected-error@*:* {{"`std::ranges::begin` is SFINAE-unfriendly on arrays of an incomplete type."}}
  // expected-error@*:* {{"`std::ranges::end` is SFINAE-unfriendly on arrays of an incomplete type."}}
  f<incomplete(&)[2][2]>();
  // expected-error@*:* {{"`std::ranges::begin` is SFINAE-unfriendly on arrays of an incomplete type."}}

  // This is okay because calling `std::ranges::end` on any rvalue is ill-formed.
  f<incomplete(&&)[10]>();
}
