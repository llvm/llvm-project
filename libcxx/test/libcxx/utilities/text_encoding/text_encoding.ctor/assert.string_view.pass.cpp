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

// text_encoding text_encoding(string_view)

#include <string_view>
#include <text_encoding>

#include "check_assertion.h"

int main(int, char**) {
  char str[std::text_encoding::max_name_length + 2] =
      "HELLOHELLOHELLOHELLOHELLOHELLOHELLOHELLOHELLOHELLOHELLOHELLOHELL";
  std::string_view view(str);

  // text_encoding(string_view) asserts if its input size() > max_name_length
  TEST_LIBCPP_ASSERT_FAILURE(std::text_encoding(view), "invalid string passed to text_encoding(string_view)");

  // text_encoding(string_view) asserts if its input contains a null terminator
  char str2[std::text_encoding::max_name_length] = "HELLOHELLOHELLOHELLOHELLOHELLOHELLOHELLOHELLOHELLOHELLOHELL";
  std::string_view view2(str2);
  str2[3] = '\0';

  TEST_LIBCPP_ASSERT_FAILURE(std::text_encoding(view2), "invalid string passed to text_encoding(string_view)");

  return 0;
}
