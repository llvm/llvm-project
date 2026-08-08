//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++11

// <memory>

#include <memory>

#include "test_macros.h"

struct Deleter {
  using pointer = long*;

  void operator()(pointer) const {}
};

void test() {
  long l = 0;
  std::unique_ptr<const int, Deleter> p{&l};

// expected-error-re@*:* {{static assertion failed due to requirement {{.+}}The returned reference must not bind to a temporary object.}}
#if TEST_STD_VER >= 26
// expected-error@*:* {{returning reference to local temporary object}}
#endif
  [[maybe_unused]] int i = *p;
}
