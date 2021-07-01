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
//           basic_string<charT,traits,Allocator>& str);

#include <string>
#include <sstream>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        std::string s("initial text");
        getline(std::istringstream(" abc\n  def\n   ghij"), s);
        assert(s == " abc");
    }
    {
        std::wstring s(L"initial text");
        getline(std::wistringstream(L" abc\n  def\n   ghij"), s);
        assert(s == L" abc");
    }
    {
        typedef std::basic_string<char, std::char_traits<char>, min_allocator<char> > S;
        S s("initial text");
        getline(std::istringstream(" abc\n  def\n   ghij"), s);
        assert(s == " abc");
    }
    {
        typedef std::basic_string<wchar_t, std::char_traits<wchar_t>, min_allocator<wchar_t> > S;
        S s(L"initial text");
        getline(std::wistringstream(L" abc\n  def\n   ghij"), s);
        assert(s == L" abc");
    }

  return 0;
}
