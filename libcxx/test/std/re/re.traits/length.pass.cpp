//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class charT> struct regex_traits;

// static std::size_t length(const char_type* p);

#include <regex>
#include <cassert>
#include "test_macros.h"

int main(int, char**)
{
    assert(std::regex_traits<char>::length("") == 0);
    assert(std::regex_traits<char>::length("1") == 1);
    assert(std::regex_traits<char>::length("12") == 2);
    assert(std::regex_traits<char>::length("123") == 3);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    assert(std::regex_traits<wchar_t>::length(L"") == 0);
    assert(std::regex_traits<wchar_t>::length(L"1") == 1);
    assert(std::regex_traits<wchar_t>::length(L"12") == 2);
    assert(std::regex_traits<wchar_t>::length(L"123") == 3);
#endif

  return 0;
}
