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

// ranges::advance
// Make sure we're SFINAE-friendly when the template argument constraints are not met.

#include <iterator>

#include <memory>

#include "test_iterators.h"

void proper_constraints() {
  auto p = std::unique_ptr<int>();
  std::ranges::advance(p, 5); // expected-error {{no matching function for call}}
  std::ranges::advance(p, p); // expected-error {{no matching function for call}}
  std::ranges::advance(p, 5, p); // expected-error {{no matching function for call}}
}
