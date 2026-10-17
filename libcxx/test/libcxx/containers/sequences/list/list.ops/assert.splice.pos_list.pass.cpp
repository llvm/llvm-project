//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}

// <list>

// void splice(const_iterator position, list& x);
//
// Calling splice with this == &x is diagnosed.

#include <list>
#include <utility>

#include "check_assertion.h"

int main(int, char**) {
  std::list<int> l{1, 2, 3};
  TEST_LIBCPP_ASSERT_FAILURE(l.splice(l.begin(), l), "list::splice(iterator, list) called with this == &list");
  TEST_LIBCPP_ASSERT_FAILURE(
      l.splice(l.begin(), std::move(l)), "list::splice(iterator, list) called with this == &list");

  return 0;
}
