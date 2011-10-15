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
//   getline(basic_istream<charT,traits>& is,
//           basic_string<charT,traits,Allocator>& str);

#include <string>
#include <sstream>
#include <cassert>

int main()
{
    {
        std::istringstream in(" abc\n  def\n   ghij");
        std::string s("initial text");
        getline(in, s);
        assert(in.good());
        assert(s == " abc");
        getline(in, s);
        assert(in.good());
        assert(s == "  def");
        getline(in, s);
        assert(in.eof());
        assert(s == "   ghij");
    }
    {
        std::wistringstream in(L" abc\n  def\n   ghij");
        std::wstring s(L"initial text");
        getline(in, s);
        assert(in.good());
        assert(s == L" abc");
        getline(in, s);
        assert(in.good());
        assert(s == L"  def");
        getline(in, s);
        assert(in.eof());
        assert(s == L"   ghij");
    }
}
