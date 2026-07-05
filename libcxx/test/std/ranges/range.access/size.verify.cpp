//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::ranges::size

#include <ranges>

extern int arr[];

// Verify that for an array of unknown bound `ranges::size` is ill-formed.
void test() {
  std::ranges::size(arr);
  // expected-error-re@-1 {{{{no matching function for call to object of type 'const (std::ranges::)?__size::__fn'}}}}
}
