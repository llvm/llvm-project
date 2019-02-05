//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>
// UNSUPPORTED: libcpp-no-exceptions
// UNSUPPORTED: c++98, c++03

// template <class BidirectionalIterator, class Allocator, class charT, class traits>
//     bool
//     regex_search(BidirectionalIterator first, BidirectionalIterator last,
//                  match_results<BidirectionalIterator, Allocator>& m,
//                  const basic_regex<charT, traits>& e,
//                  regex_constants::match_flag_type flags = regex_constants::match_default);

// Throw exception after spent too many cycles with respect to the length of the input string.

#include <regex>
#include <cassert>
#include "test_macros.h"

int main() {
  for (std::regex_constants::syntax_option_type op :
       {std::regex::ECMAScript, std::regex::extended, std::regex::egrep,
        std::regex::awk}) {
    try {
      bool b = std::regex_search(
          "aaaaaaaaaaaaaaaaaaaa",
          std::regex(
              "a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?aaaaaaaaaaaaaaaaaaaa",
              op));
      LIBCPP_ASSERT(false);
      assert(b);
    } catch (const std::regex_error &e) {
      assert(e.code() == std::regex_constants::error_complexity);
    }
  }
  std::string s(100000, 'a');
  for (std::regex_constants::syntax_option_type op :
       {std::regex::ECMAScript, std::regex::extended, std::regex::egrep,
        std::regex::awk}) {
    assert(std::regex_search(s, std::regex("a*", op)));
  }
  return 0;
}
