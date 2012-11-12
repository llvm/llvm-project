//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
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

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
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
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
