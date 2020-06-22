//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>
// UNSUPPORTED: no-exceptions
// UNSUPPORTED: c++03

// the "n" in `a{n}` should be within the numeric limits.

#include <regex>
#include <cassert>
#include "test_macros.h"

int main(int, char**) {
  for (std::regex_constants::syntax_option_type op :
       {std::regex::basic, std::regex::grep}) {
    try {
      TEST_IGNORE_NODISCARD std::regex("a\\{100000000000000000\\}", op);
      assert(false);
    } catch (const std::regex_error &e) {
      assert(e.code() == std::regex_constants::error_badbrace);
    }
  }
  for (std::regex_constants::syntax_option_type op :
       {std::regex::ECMAScript, std::regex::extended, std::regex::egrep,
        std::regex::awk}) {
    try {
      TEST_IGNORE_NODISCARD std::regex("a{100000000000000000}", op);
      assert(false);
    } catch (const std::regex_error &e) {
      assert(e.code() == std::regex_constants::error_badbrace);
    }
  }
  return 0;
}
