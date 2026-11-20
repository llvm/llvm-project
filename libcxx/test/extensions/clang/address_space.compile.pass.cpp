//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ensure that standard library facilities work with [[clang::address_space]]

// REQUIRES: std-at-least-c++20

#include <functional>

void test(int [[clang::address_space(1)]] & v) {
  // See https://llvm.org/PR243495
  std::invoke(test, v);
}
