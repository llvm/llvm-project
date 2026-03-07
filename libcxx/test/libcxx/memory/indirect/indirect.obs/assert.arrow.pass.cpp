//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// REQUIRES: has-unix-headers
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <memory>

#include "check_assertion.h"

int main(int, char**) {
  struct S {
    int n = 0;
  };
  std::indirect<S> i;
  auto(std::move(i));
  {
    TEST_LIBCPP_ASSERT_FAILURE(i->n, "operator-> called on a valueless std::indirect object");
  }
  {
    TEST_LIBCPP_ASSERT_FAILURE(std::as_const(i)->n, "operator-> called on a valueless std::indirect object");
  }

  return 0;
}
