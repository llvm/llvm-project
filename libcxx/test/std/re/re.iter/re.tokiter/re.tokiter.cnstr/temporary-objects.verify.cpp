//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Ensure that we don't allow iterators into temporary std::regex objects.

// <regex>
//
// class regex_iterator<BidirectionalIterator, charT, traits>
//
// regex_token_iterator(BidirectionalIterator a, BidirectionalIterator b,
//                      const regex_type&& re, int submatch = 0,
//                      regex_constants::match_flag_type m =
//                                              regex_constants::match_default);
//
// template <size_t N>
// regex_token_iterator(BidirectionalIterator a, BidirectionalIterator b,
//                      const regex_type&& re,
//                      const int (&submatches)[N],
//                      regex_constants::match_flag_type m =
//                                              regex_constants::match_default);
//
// regex_token_iterator(BidirectionalIterator a, BidirectionalIterator b,
//                      const regex_type&& re,
//                      initializer_list<int> submatches,
//                      regex_constants::match_flag_type m =
//                                              regex_constants::match_default);
//
// template <std::size_t N>
// regex_token_iterator(BidirectionalIterator a, BidirectionalIterator b,
//                      const regex_type&& re,
//                      const std::vector<int>& submatches,
//                      regex_constants::match_flag_type m =
//                                              regex_constants::match_default);

#include <iterator>
#include <regex>
#include <vector>

void f() {
  std::regex phone_numbers("\\d{3}-\\d{4}");
  const char phone_book[] = "start 555-1234, 555-2345, 555-3456 end";

  { // int submatch
    std::cregex_token_iterator i(std::begin(phone_book), std::end(phone_book) - 1, std::regex("\\d{3}-\\d{4}"), -1);
    // expected-error@-1 {{call to deleted constructor of 'std::cregex_token_iterator'}}
  }
  { // const int (&submatches)[N]
    const int indices[] = {-1, 0, 1};
    std::cregex_token_iterator i(
        std::begin(phone_book), std::end(phone_book) - 1, std::regex("\\d{3}-\\d{4}"), indices);
    // expected-error@-2 {{call to deleted constructor of 'std::cregex_token_iterator'}}
  }
  { // initializer_list<int> submatches
    std::cregex_token_iterator i(
        std::begin(phone_book), std::end(phone_book) - 1, std::regex("\\d{3}-\\d{4}"), {-1, 0, 1});
    // expected-error@-2 {{call to deleted constructor of 'std::cregex_token_iterator'}}
  }
  { // const std::vector<int>& submatches
    std::vector<int> v;
    v.push_back(-1);
    v.push_back(-1);
    std::cregex_token_iterator i(std::begin(phone_book), std::end(phone_book) - 1, std::regex("\\d{3}-\\d{4}"), v);
    // expected-error@-1 {{call to deleted constructor of 'std::cregex_token_iterator'}}
  }
}
