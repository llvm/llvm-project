//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// ranges::prev

#include <iterator>

#include <array>

#include "test_iterators.h"

void proper_constraints() {
  auto a = std::array{0, 1, 2};
  (void)std::ranges::prev(forward_iterator(a.begin()));    // expected-error {{no matching function for call}}
  (void)std::ranges::prev(forward_iterator(a.begin()), 5); // expected-error {{no matching function for call}}
  (void)std::ranges::prev(forward_iterator(a.begin()), 7); // expected-error {{no matching function for call}}
}
