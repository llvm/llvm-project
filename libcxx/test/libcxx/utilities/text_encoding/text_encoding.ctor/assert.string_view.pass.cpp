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

#include <string>
#include <string_view>
#include <text_encoding>

#include "check_assertion.h"

int main(int, char**) {
  std::string str('X', std::text_encoding::max_name_length + 1);

  // text_encoding(string_view) asserts if its input size() > max_name_length
  TEST_LIBCPP_ASSERT_FAILURE(std::text_encoding(std::string_view(str)), "input string_view must have size <= 63");

  // text_encoding(string_view) asserts if its input contains a null terminator
  std::string str2('X', std::text_encoding::max_name_length);
  str2[3] = '\0';

  TEST_LIBCPP_ASSERT_FAILURE(std::text_encoding(std::string_view(str2)), "input string_view must not contain '\\0'");

  return 0;
}
