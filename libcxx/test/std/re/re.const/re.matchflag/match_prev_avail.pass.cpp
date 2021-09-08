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

int main(int, char**) {
  char str1[] = "\na";
  auto str1_scnd = str1 + 1;

  // Assert that match_prev_avail disables match_not_bol and this matches
  assert(std::regex_match(str1 + 1, str1 + 2, std::regex("^a"),
                     std::regex_constants::match_not_bol |
                         std::regex_constants::match_prev_avail));
  // Manually passing match_prev_avail defines that --str1 is a valid position
  assert(std::regex_match(str1_scnd, std::regex("a"),
                     std::regex_constants::match_not_bol |
                         std::regex_constants::match_prev_avail));

  //Assert that match_prev_avail disables match_not_bow and this matches
  assert(std::regex_search(str1, std::regex("\\ba")));
  assert(std::regex_match(str1 + 1, str1 + 2, std::regex("\\ba\\b"),
                     std::regex_constants::match_not_bow |
                         std::regex_constants::match_prev_avail));
  assert(std::regex_search(str1_scnd, std::regex("\\ba"),
                      std::regex_constants::match_not_bow |
                          std::regex_constants::match_prev_avail));

  //Assert that match_prev_avail disables both match_not_bow and match_not_bol
  assert(std::regex_match(str1 + 1, str1 + 2, std::regex("^a"),
                     std::regex_constants::match_not_bol |
                         std::regex_constants::match_not_bow |
                         std::regex_constants::match_prev_avail));
  assert(std::regex_match(str1_scnd, std::regex("\\ba"),
                     std::regex_constants::match_not_bol |
                         std::regex_constants::match_not_bow |
                         std::regex_constants::match_prev_avail));

  // pr 42199
  std::string S = " cd";
  std::string::iterator Start = S.begin() + 1;
  std::string::iterator End = S.end();
  assert(std::regex_search(Start, End, std::regex("^cd")));

  assert(!std::regex_search(Start, End, std::regex("^cd"),
            std::regex_constants::match_not_bol));
  assert(!std::regex_search(Start, End, std::regex(".*\\bcd\\b"),
            std::regex_constants::match_not_bow));
  assert(!std::regex_search(Start, End, std::regex("^cd"),
            std::regex_constants::match_not_bol |
            std::regex_constants::match_not_bow));
  assert(!std::regex_search(Start, End, std::regex(".*\\bcd\\b"),
            std::regex_constants::match_not_bol |
            std::regex_constants::match_not_bow));

  assert(std::regex_search(Start, End, std::regex("^cd"),
            std::regex_constants::match_prev_avail));

  assert(std::regex_search(Start, End, std::regex("^cd"),
            std::regex_constants::match_not_bol |
            std::regex_constants::match_prev_avail));
  assert(std::regex_search(Start, End, std::regex("^cd"),
            std::regex_constants::match_not_bow |
            std::regex_constants::match_prev_avail));
  assert(std::regex_match(Start, End, std::regex("\\bcd\\b"),
            std::regex_constants::match_not_bol |
            std::regex_constants::match_not_bow |
            std::regex_constants::match_prev_avail));
  return 0;
}
