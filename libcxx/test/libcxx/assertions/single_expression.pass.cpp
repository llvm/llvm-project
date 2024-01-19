//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that `_LIBCPP_ASSERT` and `_LIBCPP_ASSUME` are each a single expression, and that it still holds when an
// assertion is wrapped in  the `_LIBCPP_REDUNDANT_ASSERTION` macro. This is useful so we can use them in places that
// require an expression, such as in a constructor initializer list.

#include <__assert>
#include <cassert>

void test_assert() {
  int i = (_LIBCPP_ASSERT(true, "message"), 3);
  assert(i == 3);
  return _LIBCPP_ASSERT(true, "message");
}

void test_assume() {
  int i = (_LIBCPP_ASSUME(true), 3);
  assert(i == 3);
  return _LIBCPP_ASSUME(true);
}

void test_redundant() {
  int i = (_LIBCPP_REDUNDANT_ASSERTION(_LIBCPP_ASSERT(true, "message")), 3);
  assert(i == 3);
  return _LIBCPP_REDUNDANT_ASSERTION(_LIBCPP_ASSERT(true, "message"));
}

int main(int, char**) {
  test_assert();
  test_assume();
  test_redundant();

  return 0;
}
