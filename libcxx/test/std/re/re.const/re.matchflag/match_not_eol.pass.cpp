//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// match_not_eol:
//     The last character in the sequence [first,last) shall be treated as
//     though it is not at the end of a line, so the character "$" in
//     the regular expression shall not match [last,last).

#include <regex>
#include <cassert>
#include "test_macros.h"

int main(int, char**)
{
    {
    std::string target = "foo";
    std::regex re("foo$");
    assert( std::regex_match(target, re));
    assert(!std::regex_match(target, re, std::regex_constants::match_not_eol));
    }

    {
    std::string target = "foo";
    std::regex re("foo");
    assert( std::regex_match(target, re));
    assert( std::regex_match(target, re, std::regex_constants::match_not_eol));
    }

    {
    std::string target = "refoo";
    std::regex re("foo$");
    assert( std::regex_search(target, re));
    assert(!std::regex_search(target, re, std::regex_constants::match_not_eol));
    }

    {
    std::string target = "refoo";
    std::regex re("foo");
    assert( std::regex_search(target, re));
    assert( std::regex_search(target, re, std::regex_constants::match_not_eol));
    }

    {
      std::string target = "foo";
      std::regex re("$");
      assert(std::regex_search(target, re));
      assert(!std::regex_search(target, re, std::regex_constants::match_not_eol));

      std::smatch match;
      assert(std::regex_search(target, match, re));
      assert(match.position(0) == 3);
      assert(match.length(0) == 0);
      assert(!std::regex_search(target, match, re, std::regex_constants::match_not_eol));
      assert(match.length(0) == 0);
    }

    {
      std::string target = "foo";
      std::regex re("$", std::regex::multiline);
      std::smatch match;
      assert(std::regex_search(target, match, re));
      assert(match.position(0) == 3);
      assert(match.length(0) == 0);
      assert(!std::regex_search(target, match, re, std::regex_constants::match_not_eol));
      assert(match.length(0) == 0);
    }

    {
      std::string target = "foo";
      std::regex re("$");
      assert(!std::regex_match(target, re));
      assert(!std::regex_match(target, re, std::regex_constants::match_not_eol));
    }

    {
      std::string target = "a";
      std::regex re("^b*$");
      assert(!std::regex_search(target, re));
      assert(!std::regex_search(target, re, std::regex_constants::match_not_eol));
    }

  return 0;
}
