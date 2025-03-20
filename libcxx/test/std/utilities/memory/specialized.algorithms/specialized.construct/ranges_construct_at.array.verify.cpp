//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// REQUIRES: stdlib=libc++

// <memory>

// Test that std::ranges::construct_at provides a meaningful diagnostic when used with an
// array type and construction arguments are provided. See LWG3436.

#include <memory>

using Array = int[3];
void test(Array* a) {
  std::ranges::construct_at(a, 1, 2, 3);
  // expected-error-re@*:* {{static assertion failed {{.*}}construction arguments cannot be passed to construct_at with an array type}}
}
