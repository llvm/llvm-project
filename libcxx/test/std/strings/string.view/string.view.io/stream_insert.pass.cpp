//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits, class Allocator>
//   basic_ostream<charT, traits>&
//   operator<<(basic_ostream<charT, traits>& os,
//              const basic_string_view<charT,traits> str);

#include <string_view>
#include <sstream>
#include <cassert>

using std::string_view;
using std::wstring_view;

int main(int, char**)
{
    {
        std::ostringstream out;
        string_view sv("some text");
        out << sv;
        assert(out.good());
        assert(sv == out.str());
    }
    {
        std::ostringstream out;
        std::string s("some text");
        string_view sv(s);
        out.width(12);
        out << sv;
        assert(out.good());
        assert("   " + s == out.str());
    }
    {
        std::wostringstream out;
        wstring_view sv(L"some text");
        out << sv;
        assert(out.good());
        assert(sv == out.str());
    }
    {
        std::wostringstream out;
        std::wstring s(L"some text");
        wstring_view sv(s);
        out.width(12);
        out << sv;
        assert(out.good());
        assert(L"   " + s == out.str());
    }

  return 0;
}
