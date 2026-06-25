//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// std::ranges::reserve_hint

#include <ranges>

extern int arr[];

// Verify that for an array of unknown bound `ranges::reserve_hint` is ill-formed.
void test() {
  std::ranges::reserve_hint(arr);
  // expected-error-re@-1 {{{{no matching function for call to object of type 'const (std::ranges::)?__reserve_hint::__fn'}}}}
}
