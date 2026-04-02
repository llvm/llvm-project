//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// Check that functions are marked [[nodiscard]]

#include <ranges>
#include <vector>

void test() {
  std::vector<int> range;
  std::allocator<int> alloc;

  { // `ranges::to` base template -- the `_Container` type is a simple type template parameter.

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::to<std::vector<int>>(range);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::to<std::vector<int>>(range, alloc);
  }

  { // `ranges::to` specialization -- `_Container` is a template template parameter requiring deduction to figure out the
    // container element type.

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::to<std::vector>(range);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::to<std::vector>(range, alloc);
  }

  { // Range adaptor closure object 1 -- wrapping the `ranges::to` version where `_Container` is a simple type template
    // parameter.

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::to<std::vector<int>>();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::to<std::vector<int>>(alloc);
  }

  { // Range adaptor closure object 2 -- wrapping the `ranges::to` version where `_Container` is a template template
    // parameter.

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::to<std::vector>();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::to<std::vector>(alloc);
  }
}
