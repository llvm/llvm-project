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

#include <cassert>
#include <regex>
using namespace std;

int main(int, char**) {
  char str1[] = "\na";
  auto str1_scnd = str1 + 1;
  // Assert that match_prev_avail disables match_not_bol and this matches
  assert(regex_match(str1 + 1, str1 + 2, regex("^a"),
                     regex_constants::match_not_bol |
                         regex_constants::match_prev_avail));
  // Manually passing match_prev_avail defines that --str1 is a valid position
  assert(regex_match(str1_scnd, regex("a"),
                     regex_constants::match_not_bol |
                         regex_constants::match_prev_avail));

  //Assert that match_prev_avail disables match_not_bow and this matches
  assert(regex_search(str1, regex("\\ba")));
  assert(regex_match(str1 + 1, str1 + 2, regex("\\ba\\b"),
                     regex_constants::match_not_bow |
                         regex_constants::match_prev_avail));
  assert(regex_search(str1_scnd, regex("\\ba"),
                      regex_constants::match_not_bow |
                          regex_constants::match_prev_avail));

  //Assert that match_prev_avail disables both match_not_bow and match_not_bol
  assert(regex_match(str1 + 1, str1 + 2, regex("^a"),
                     regex_constants::match_not_bol |
                         regex_constants::match_not_bow |
                         regex_constants::match_prev_avail));
  assert(regex_match(str1_scnd, regex("\\ba"),
                     regex_constants::match_not_bol |
                         regex_constants::match_not_bow |
                         regex_constants::match_prev_avail));

  // pr 42199
  string S = " cd";
  string::iterator Start = S.begin() + 1;
  string::iterator End = S.end();
  assert(regex_search(Start, End, regex("^cd")));

  assert(
      !regex_search(Start, End, regex("^cd"), regex_constants::match_not_bol));
  assert(!regex_search(Start, End, regex(".*\\bcd\\b"),
                       regex_constants::match_not_bow));
  assert(!regex_search(Start, End, regex("^cd"),
                       regex_constants::match_not_bol |
                           regex_constants::match_not_bow));
  assert(!regex_search(Start, End, regex(".*\\bcd\\b"),
                       regex_constants::match_not_bol |
                           regex_constants::match_not_bow));

  assert(regex_search(Start, End, regex("^cd"),
                      regex_constants::match_prev_avail));

  assert(regex_search(Start, End, regex("^cd"),
                      regex_constants::match_not_bol |
                          regex_constants::match_prev_avail));
  assert(regex_search(Start, End, regex("^cd"),
                      regex_constants::match_not_bow |
                          regex_constants::match_prev_avail));
  assert(regex_match(Start, End, regex("\\bcd\\b"),
                     regex_constants::match_not_bol |
                         regex_constants::match_not_bow |
                         regex_constants::match_prev_avail));
  return 0;
}
