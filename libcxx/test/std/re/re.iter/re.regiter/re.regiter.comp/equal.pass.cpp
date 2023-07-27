//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <regex>

// class regex_token_iterator<BidirectionalIterator, charT, traits>

// bool operator==(default_sentinel_t) const { return *this == regex_iterator(); } // since C++20

#include <cassert>
#include <iterator>
#include <regex>

#include "test_comparisons.h"

int main(int, char**) {
  AssertEqualityReturnBool<std::cregex_iterator>();

  {
    std::cregex_iterator i;
    assert(testEquality(i, std::default_sentinel, true));
  }

  AssertEqualityReturnBool<std::sregex_token_iterator>();

  {
    std::sregex_token_iterator i;
    assert(testEquality(i, std::default_sentinel, true));
  }

  return 0;
}
