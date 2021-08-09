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

// std::ranges::data

#include <ranges>

struct Incomplete;

void f(Incomplete arr[]) {
  // expected-error@*:* {{is SFINAE-unfriendly on arrays of an incomplete type.}}
  // expected-error@*:* {{no matching function for call}}
  std::ranges::data(arr);
}

void f(Incomplete(&arr)[]) {
  // expected-error@*:* {{is SFINAE-unfriendly on arrays of an incomplete type.}}
  // expected-error@*:* {{no matching function for call}}
  std::ranges::data(arr);
}

void f(Incomplete(&&arr)[]) {
  // expected-error@*:* {{is SFINAE-unfriendly on arrays of an incomplete type.}}
  // expected-error@*:* {{no matching function for call}}
  std::ranges::data(arr);
}

void f2(Incomplete arr[2]) {
  // expected-error@*:* {{no matching function for call}}
  std::ranges::data(arr);
}

void f(Incomplete(&arr)[2]) {
  // expected-error@*:* {{no matching function for call}}
  std::ranges::data(arr);
}

void f(Incomplete(&&arr)[2]) {
  // expected-error@*:* {{no matching function for call}}
  std::ranges::data(arr);
}

void f(Incomplete(&arr)[2][2]) {
  // expected-error@*:* {{no matching function for call}}
  std::ranges::data(arr);
}
