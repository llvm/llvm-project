//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <regex>

// template <class BidirectionalIterator, class Allocator, class charT, class traits>
//     bool
//     regex_search(BidirectionalIterator first, BidirectionalIterator last,
//                  match_results<BidirectionalIterator, Allocator>& m,
//                  const basic_regex<charT, traits>& e,
//                  regex_constants::match_flag_type flags = regex_constants::match_default);

#include <regex>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

int main()
{
    {
        std::cmatch m;
        const char s[] = "tournament";
        assert(std::regex_search(s, m, std::regex("tour\nto\ntournament",
                std::regex_constants::egrep)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s + std::char_traits<char>::length(s));
        assert(m.length(0) == 10);
        assert(m.position(0) == 0);
        assert(m.str(0) == "tournament");
    }
    {
        std::cmatch m;
        const char s[] = "ment";
        assert(std::regex_search(s, m, std::regex("tour\n\ntournament",
                std::regex_constants::egrep)));
        assert(m.size() == 1);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s + std::char_traits<char>::length(s));
        assert(m.length(0) == 0);
        assert(m.position(0) == 0);
        assert(m.str(0) == "");
    }
    {
        std::cmatch m;
        const char s[] = "tournament";
        assert(std::regex_search(s, m, std::regex("(tour|to|tournament)+\ntourna",
                std::regex_constants::egrep)));
        assert(m.size() == 2);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s + std::char_traits<char>::length(s));
        assert(m.length(0) == 10);
        assert(m.position(0) == 0);
        assert(m.str(0) == "tournament");
    }
    {
        std::cmatch m;
        const char s[] = "tourna";
        assert(std::regex_search(s, m, std::regex("(tour|to|tournament)+\ntourna",
                std::regex_constants::egrep)));
        assert(m.size() == 2);
        assert(!m.prefix().matched);
        assert(m.prefix().first == s);
        assert(m.prefix().second == m[0].first);
        assert(!m.suffix().matched);
        assert(m.suffix().first == m[0].second);
        assert(m.suffix().second == s + std::char_traits<char>::length(s));
        assert(m.length(0) == 6);
        assert(m.position(0) == 0);
        assert(m.str(0) == "tourna");
    }
}
