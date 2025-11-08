//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <span>

// Check that functions are marked [[nodiscard]]

#include <array>
#include <span>
#include <vector>

void test() {
  { // Test with static extent
    std::array arr{94, 92};
    std::span sp{arr};
    sp.at(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
  { // Test with dynamic extent
    std::vector vec{94, 92};
    std::span sp{vec};
    sp.at(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  }
}
