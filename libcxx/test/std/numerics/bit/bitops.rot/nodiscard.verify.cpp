//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Check that std::rotl and std::rotr are marked [[nodiscard]]

#include <bit>

void func() {
  std::rotl(0u, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::rotr(0u, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
