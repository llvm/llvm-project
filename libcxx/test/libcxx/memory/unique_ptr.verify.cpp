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

int main(int, char**) {
  long l = 0;
  std::unique_ptr<const int, deleter> p(&l);
  int i =
      *p; // expected-error@*:* {{static assertion failed due to requirement '!__reference_converts_from_temporary(const int &, long &)': Reference type _Tp must not convert from a temporary object}}
  return 0;
}
