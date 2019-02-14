//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// class match_results<BidirectionalIterator, Allocator>

// string_type
//   format(const char_type* fmt,
//          regex_constants::match_flag_type flags = regex_constants::format_default) const;

#include <iostream>

#include <regex>
#include <cassert>
#include "test_macros.h"

int main(int, char**)
{
    {
        std::match_results<const char*> m;
        const char s[] = "abcdefghijk";
        assert(std::regex_search(s, m, std::regex("cd((e)fg)hi")));

        const char fmt[] = "prefix: $`, match: $&, suffix: $', m[1]: $1, m[2]: $2";
        std::string out = m.format(fmt);
        assert(out == "prefix: ab, match: cdefghi, suffix: jk, m[1]: efg, m[2]: e");
    }
    {
        std::match_results<const char*> m;
        const char s[] = "abcdefghijk";
        assert(std::regex_search(s, m, std::regex("cd((e)fg)hi")));

        const char fmt[] = "prefix: $`, match: $&, suffix: $', m[1]: $1, m[2]: $2";
        std::string out = m.format(fmt, std::regex_constants::format_sed);
        assert(out == "prefix: $`, match: $cdefghi, suffix: $', m[1]: $1, m[2]: $2");
    }
    {
        std::match_results<const char*> m;
        const char s[] = "abcdefghijk";
        assert(std::regex_search(s, m, std::regex("cd((e)fg)hi")));

        const char fmt[] = "match: &, m[1]: \\1, m[2]: \\2";
        std::string out = m.format(fmt, std::regex_constants::format_sed);
        assert(out == "match: cdefghi, m[1]: efg, m[2]: e");
    }

    {
        std::match_results<const wchar_t*> m;
        const wchar_t s[] = L"abcdefghijk";
        assert(std::regex_search(s, m, std::wregex(L"cd((e)fg)hi")));

        const wchar_t fmt[] = L"prefix: $`, match: $&, suffix: $', m[1]: $1, m[2]: $2";
        std::wstring out = m.format(fmt);
        assert(out == L"prefix: ab, match: cdefghi, suffix: jk, m[1]: efg, m[2]: e");
    }
    {
        std::match_results<const wchar_t*> m;
        const wchar_t s[] = L"abcdefghijk";
        assert(std::regex_search(s, m, std::wregex(L"cd((e)fg)hi")));

        const wchar_t fmt[] = L"prefix: $`, match: $&, suffix: $', m[1]: $1, m[2]: $2";
        std::wstring out = m.format(fmt, std::regex_constants::format_sed);
        assert(out == L"prefix: $`, match: $cdefghi, suffix: $', m[1]: $1, m[2]: $2");
    }
    {
        std::match_results<const wchar_t*> m;
        const wchar_t s[] = L"abcdefghijk";
        assert(std::regex_search(s, m, std::wregex(L"cd((e)fg)hi")));

        const wchar_t fmt[] = L"match: &, m[1]: \\1, m[2]: \\2";
        std::wstring out = m.format(fmt, std::regex_constants::format_sed);
        assert(out == L"match: cdefghi, m[1]: efg, m[2]: e");
    }

  return 0;
}
