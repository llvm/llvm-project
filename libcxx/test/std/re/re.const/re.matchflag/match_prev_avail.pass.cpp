//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// match_prev_avail:
//     --first is a valid iterator position. When this flag is set the flags
//     match_not_bol and match_not_bow shall be ignored by the regular
//     expression algorithms (30.11) and iterators (30.12)

#include <regex>

#include <cassert>
#include <string>

template <class It>
void test(It start,
          It end,
          char const* regex,
          std::regex_constants::match_flag_type flags,
          bool expect_match,
          int expect_pos = 0,
          int expect_len = 0,
          bool multiline = false) {
  std::smatch match;
  std::regex re(regex, multiline ? std::regex::multiline : std::regex::ECMAScript);
  if (expect_match) {
    assert(std::regex_search(start, end, match, re, flags));
    assert(match.position(0) == expect_pos);
    assert(match.length(0) == expect_len);
  } else {
    assert(!std::regex_search(start, end, match, re, flags));
  }
}

int main(int, char**) {
  // The implementation of `match_prev_avail` is being corrected as per the discussions in the issue #74838.
  {
    std::string s = "ab";
    test(s.cbegin() + 1, s.cend(), "^", std::regex_constants::match_default, true, 0, 0);
    test(s.cbegin() + 1, s.cend(), "^", std::regex_constants::match_not_bol, false);
    test(s.cbegin() + 1, s.cend(), "^", std::regex_constants::match_prev_avail, false);
    test(s.cbegin() + 1,
         s.cend(),
         "^",
         std::regex_constants::match_prev_avail | std::regex_constants::match_not_bol,
         false);
  }

  {
    std::string s = "ab";
    test(s.cbegin(), s.cend(), "^ab", std::regex_constants::match_default, true, 0, 2);
    test(s.cbegin(), s.cend(), "^ab", std::regex_constants::match_not_bol, false);
  }

  {
    std::string s = "ab";
    test(s.cbegin() + 1, s.cend(), "^b", std::regex_constants::match_default, true, 0, 1);
    test(s.cbegin() + 1, s.cend(), "^b", std::regex_constants::match_not_bol, false);
    test(s.cbegin() + 1, s.cend(), "^b", std::regex_constants::match_prev_avail, false);
    test(s.cbegin() + 1,
         s.cend(),
         "^b",
         std::regex_constants::match_prev_avail | std::regex_constants::match_not_bol,
         false);
  }

  {
    std::string s = "ab\nb";
    test(s.cbegin() + 1, s.cend(), "^b", std::regex_constants::match_default, true, 0, 1, true);
    test(s.cbegin() + 1, s.cend(), "^b", std::regex_constants::match_not_bol, true, 2, 1, true); // TODO
    test(s.cbegin() + 1, s.cend(), "^b", std::regex_constants::match_prev_avail, true, 2, 1, true);
    test(s.cbegin() + 1,
         s.cend(),
         "^b",
         std::regex_constants::match_prev_avail | std::regex_constants::match_not_bol,
         true,
         2,
         1,
         true);
  }

  {
    std::string s = "\na";
    test(s.cbegin() + 1,
         s.cend(),
         "^a",
         std::regex_constants::match_not_bol | std::regex_constants::match_prev_avail,
         false);
    test(s.cbegin() + 1,
         s.cend(),
         "a",
         std::regex_constants::match_not_bol | std::regex_constants::match_prev_avail,
         true,
         0,
         1);

    test(s.cbegin() + 1,
         s.cend(),
         "\\ba",
         std::regex_constants::match_not_bow | std::regex_constants::match_prev_avail,
         true,
         0,
         1);
    test(s.cbegin() + 1,
         s.cend(),
         "\\ba\\b",
         std::regex_constants::match_not_bow | std::regex_constants::match_prev_avail,
         true,
         0,
         1);

    test(s.cbegin() + 1,
         s.cend(),
         "^a",
         std::regex_constants::match_not_bol | std::regex_constants::match_not_bow |
             std::regex_constants::match_prev_avail,
         false);
    test(s.cbegin() + 1,
         s.cend(),
         "\\ba",
         std::regex_constants::match_not_bol | std::regex_constants::match_not_bow |
             std::regex_constants::match_prev_avail,
         true,
         0,
         1);
  }

  {
    // pr 42199
    std::string s = " cd";
    test(s.cbegin() + 1, s.cend(), "^cd", std::regex_constants::match_default, true, 0, 2);
    test(s.cbegin() + 1, s.cend(), "^cd", std::regex_constants::match_not_bol, false);
    test(s.cbegin() + 1, s.cend(), ".*\\bcd\\b", std::regex_constants::match_not_bow, false);
    test(s.cbegin() + 1,
         s.cend(),
         "^cd",
         std::regex_constants::match_not_bol | std::regex_constants::match_not_bow,
         false);
    test(s.cbegin() + 1,
         s.cend(),
         ".*\\bcd\\b",
         std::regex_constants::match_not_bol | std::regex_constants::match_not_bow,
         false);

    test(s.cbegin() + 1, s.cend(), "^cd", std::regex_constants::match_prev_avail, false);
    test(s.cbegin() + 1,
         s.cend(),
         "^cd",
         std::regex_constants::match_not_bol | std::regex_constants::match_prev_avail,
         false);
    test(s.cbegin() + 1,
         s.cend(),
         "^cd",
         std::regex_constants::match_not_bow | std::regex_constants::match_prev_avail,
         false);
    test(s.cbegin() + 1,
         s.cend(),
         "\\bcd\\b",
         std::regex_constants::match_not_bol | std::regex_constants::match_not_bow |
             std::regex_constants::match_prev_avail,
         true,
         0,
         2);
  }
  return 0;
}