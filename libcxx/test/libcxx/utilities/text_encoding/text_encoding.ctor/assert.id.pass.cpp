//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}

// <text_encoding>

// text_encoding text_encoding(id)

#include <climits>
#include <text_encoding>

#include "check_assertion.h"

int main(int, char**) {
  // Make sure that text_encoding(id) asserts when the input id not in range of allowed values

  TEST_LIBCPP_ASSERT_FAILURE(
      std::text_encoding(std::text_encoding::id(33)), "invalid text_encoding::id passed to text_encoding(id)");
  TEST_LIBCPP_ASSERT_FAILURE(
      std::text_encoding(std::text_encoding::id(34)), "invalid text_encoding::id passed to text_encoding(id)");
  TEST_LIBCPP_ASSERT_FAILURE(
      std::text_encoding(std::text_encoding::id(-1)), "invalid text_encoding::id passed to text_encoding(id)");
  TEST_LIBCPP_ASSERT_FAILURE(
      std::text_encoding(std::text_encoding::id(INT_MAX)), "invalid text_encoding::id passed to text_encoding(id)");

  return 0;
}
