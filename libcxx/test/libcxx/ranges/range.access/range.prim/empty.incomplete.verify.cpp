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

// Test the libc++ specific behavior that we provide a better diagnostic when calling
// std::ranges::empty on an array of incomplete type.

#include <ranges>

struct Incomplete;

void f(Incomplete arr[]) {
  // expected-error@*:* {{is SFINAE-unfriendly on arrays of an incomplete type.}}
  // expected-error@*:* {{call to deleted function call operator in type}}
  // expected-error@*:* {{attempt to use a deleted function}}
  std::ranges::begin(arr);
}

void f(Incomplete(&arr)[]) {
  // expected-error@*:* {{is SFINAE-unfriendly on arrays of an incomplete type.}}
  std::ranges::begin(arr);
}

void f(Incomplete(&&arr)[]) {
  // expected-error@*:* {{is SFINAE-unfriendly on arrays of an incomplete type.}}
  std::ranges::begin(arr);
}

void f2(Incomplete arr[2]) {
  // expected-error@*:* {{call to deleted function call operator in type}}
  // expected-error@*:* {{attempt to use a deleted function}}
  std::ranges::begin(arr);
}

void f(Incomplete(&arr)[2]) {
  std::ranges::begin(arr);
}

void f(Incomplete(&&arr)[2]) {
  std::ranges::begin(arr);
}

void f(Incomplete(&arr)[2][2]) {
  std::ranges::begin(arr);
}
