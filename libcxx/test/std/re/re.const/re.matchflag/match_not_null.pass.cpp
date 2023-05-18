//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// match_not_null:
//     The regular expression shall not match an empty sequence.

#include "test_macros.h"
#include <cassert>
#include <regex>

int main(int, char**)
{
  // When match_not_null is on, the regex engine should reject empty matches and
  // move on to try other solutions.
  std::cmatch m;
  assert(!std::regex_search("a", m, std::regex("b*"),
                            std::regex_constants::match_not_null));
  assert(std::regex_search("aa", m, std::regex("a*?"),
                           std::regex_constants::match_not_null));
  assert(m[0].length() == 1);
  assert(!std::regex_search("a", m, std::regex("b*", std::regex::extended),
                            std::regex_constants::match_not_null));
  assert(!std::regex_search(
      "a", m,
      std::regex("b*", std::regex::extended | std::regex_constants::nosubs),
      std::regex_constants::match_not_null));

  assert(!std::regex_match("", m, std::regex("a*"),
                           std::regex_constants::match_not_null));
  assert(!std::regex_match("", m, std::regex("a*", std::regex::extended),
                           std::regex_constants::match_not_null));
  assert(!std::regex_match(
      "", m,
      std::regex("a*", std::regex::extended | std::regex_constants::nosubs),
      std::regex_constants::match_not_null));

  return 0;
}
