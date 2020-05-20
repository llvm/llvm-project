//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>
// UNSUPPORTED: no-exceptions
// UNSUPPORTED: c++98, c++03

// the "n" and "m" in `a{n,m}` should be within the numeric limits.
// requirement "m >= n" should be checked.

#include <regex>
#include <cassert>
#include "test_macros.h"

int main(int, char**) {
  // test that `n <= m`
  for (std::regex_constants::syntax_option_type op :
       {std::regex::basic}) {
    try {
      TEST_IGNORE_NODISCARD std::regex("a\\{3,2\\}", op);
      assert(false);
    } catch (const std::regex_error &e) {
      assert(e.code() == std::regex_constants::error_badbrace);
      LIBCPP_ASSERT(e.code() == std::regex_constants::error_badbrace);
    }
  }
  for (std::regex_constants::syntax_option_type op :
       {std::regex::ECMAScript, std::regex::extended, std::regex::egrep,
        std::regex::awk}) {
    try {
      TEST_IGNORE_NODISCARD std::regex("a{3,2}", op);
      assert(false);
    } catch (const std::regex_error &e) {
      assert(e.code() == std::regex_constants::error_badbrace);
      LIBCPP_ASSERT(e.code() == std::regex_constants::error_badbrace);
    }
  }

  // test that both bounds are within the limit
  for (std::regex_constants::syntax_option_type op :
       {std::regex::basic}) {
    try {
      TEST_IGNORE_NODISCARD std::regex("a\\{100000000000000000000,10000000000000000000\\}", op);
      assert(false);
    } catch (const std::regex_error &e) {
      assert(e.code() == std::regex_constants::error_badbrace);
      LIBCPP_ASSERT(e.code() == std::regex_constants::error_badbrace);
    }
  }
  for (std::regex_constants::syntax_option_type op :
       {std::regex::ECMAScript, std::regex::extended, std::regex::egrep,
        std::regex::awk}) {
    try {
      TEST_IGNORE_NODISCARD std::regex("a{100000000000000000000,10000000000000000000}", op);
      assert(false);
    } catch (const std::regex_error &e) {
      assert(e.code() == std::regex_constants::error_badbrace);
      LIBCPP_ASSERT(e.code() == std::regex_constants::error_badbrace);
    }
  }
  return 0;
}
