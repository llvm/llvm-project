//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that _LIBCPP_ASSERT is a single expression. This is useful so we can use
// it in places that require an expression, such as in a constructor initializer list.

// RUN: %{build} -Wno-macro-redefined -D_LIBCPP_ENABLE_ASSERTIONS=1
// RUN: %{run}

// RUN: %{build} -Wno-macro-redefined -D_LIBCPP_ENABLE_ASSERTIONS=0
// RUN: %{run}

// RUN: %{build} -Wno-macro-redefined -D_LIBCPP_ENABLE_ASSERTIONS=0 -D_LIBCPP_ASSERTIONS_DISABLE_ASSUME
// RUN: %{run}

#include <__assert>
#include <cassert>

void f() {
  int i = (_LIBCPP_ASSERT(true, "message"), 3);
  assert(i == 3);
  return _LIBCPP_ASSERT(true, "message");
}

int main(int, char**) {
  f();
  return 0;
}
