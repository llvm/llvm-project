//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// views::iota

#include <ranges>

#include "types.h"

void test() {
  // LWG4096
  {
    [[maybe_unused]] auto i1 = std::views::iota(0); // OK
    [[maybe_unused]] auto i2 = std::views::iota(std::views::iota(0));
    // expected-error@*:* {{no matching function for call to object of type 'const __iota::__fn'}}
  }
  {
    [[maybe_unused]] auto i1 = std::views::iota(SomeInt(0)); // OK
    [[maybe_unused]] auto i2 = std::views::iota(std::views::iota(SomeInt(0)));
    //expected-error@*:* {{no matching function for call to object of type 'const __iota::__fn'}}
  }
}
