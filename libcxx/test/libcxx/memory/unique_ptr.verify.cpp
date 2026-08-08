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

struct deleter {
  using pointer = long*;
  void operator()(pointer) const {}
};

void test() {
  long l = 0;
  std::unique_ptr<const int, deleter> p(&l);
// expected-error-re@*:* {{static assertion failed{{.*}}the returned reference must not bind to a temporary object}}
  // expected-error@*:* 0-1{{returning reference to local temporary object}}
  (void)*p; // expected-note {{requested here}}
}
