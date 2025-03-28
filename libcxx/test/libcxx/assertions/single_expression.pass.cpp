//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that `_LIBCPP_ASSERT` and `_LIBCPP_ASSUME` are each a single expression.
// This is useful so we can use them  in places that require an expression, such as
// in a constructor initializer list.

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <__assert>
#include <cassert>

void f() {
  int i = (_LIBCPP_ASSERT(true, "message"), 3);
  assert(i == 3);
  return _LIBCPP_ASSERT(true, "message");
}

void g() {
  int i = (_LIBCPP_ASSUME(true), 3);
  assert(i == 3);
  return _LIBCPP_ASSUME(true);
}

int main(int, char**) {
  f();
  g();
  return 0;
}
