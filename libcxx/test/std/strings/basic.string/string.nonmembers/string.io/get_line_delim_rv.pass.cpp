//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   basic_istream<charT,traits>&
//   getline(basic_istream<charT,traits>&& is,
//           basic_string<charT,traits,Allocator>& str, charT delim);

#include <string>
#include <sstream>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        std::string s("initial text");
        std::getline(std::istringstream(" abc*  def*   ghij"), s, '*');
        assert(s == " abc");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::wstring s(L"initial text");
        std::getline(std::wistringstream(L" abc*  def*   ghij"), s, L'*');
        assert(s == L" abc");
    }
#endif
    {
        typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > S;
        S s("initial text");
        std::getline(std::istringstream(" abc*  def*   ghij"), s, '*');
        assert(s == " abc");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        typedef std::basic_string<wchar_t, std::char_traits<wchar_t>, min_allocator<wchar_t> > S;
        S s(L"initial text");
        std::getline(std::wistringstream(L" abc*  def*   ghij"), s, L'*');
        assert(s == L" abc");
    }
#endif

  return 0;
}
