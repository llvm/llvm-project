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
//   basic_ostream<charT, traits>&
//   operator<<(basic_ostream<charT, traits>& os,
//              const basic_string<charT,traits,Allocator>& str);

#include <string>
#include <sstream>
#include <cassert>

int main()
{
    {
        std::ostringstream out;
        std::string s("some text");
        out << s;
        assert(out.good());
        assert(s == out.str());
    }
    {
        std::ostringstream out;
        std::string s("some text");
        out.width(12);
        out << s;
        assert(out.good());
        assert("   " + s == out.str());
    }
    {
        std::wostringstream out;
        std::wstring s(L"some text");
        out << s;
        assert(out.good());
        assert(s == out.str());
    }
    {
        std::wostringstream out;
        std::wstring s(L"some text");
        out.width(12);
        out << s;
        assert(out.good());
        assert(L"   " + s == out.str());
    }
}
