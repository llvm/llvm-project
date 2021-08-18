//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: clang-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// unspecified begin;

#include <ranges>

#include <type_traits>

using cend_t = decltype(std::ranges::cend);

// clang-format off
template <class T>
requires(!std::invocable<cend_t&, T>)
void f() {}
// clang-format on

void test() {
  struct incomplete;
  f<incomplete(&)[]>();
  // expected-error@*:* {{"`std::ranges::begin` is SFINAE-unfriendly on arrays of an incomplete type."}}
  // expected-error@*:* {{"`std::ranges::end` is SFINAE-unfriendly on arrays of an incomplete type."}}
  f<incomplete(&)[10]>();
  // expected-error@*:* {{"`std::ranges::begin` is SFINAE-unfriendly on arrays of an incomplete type."}}
  // expected-error@*:* {{"`std::ranges::end` is SFINAE-unfriendly on arrays of an incomplete type."}}
  // expected-error@-3 {{no matching function for call to 'f'}}
  f<incomplete(&)[2][2]>();
  // expected-error@*:* {{"`std::ranges::begin` is SFINAE-unfriendly on arrays of an incomplete type."}}
  // expected-error@-2 {{no matching function for call to 'f'}}

  // This is okay because calling `std::ranges::end` on any rvalue is ill-formed.
  f<incomplete(&&)[10]>();
}
