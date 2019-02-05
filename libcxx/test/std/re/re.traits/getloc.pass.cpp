//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8

// <regex>

// template <class charT> struct regex_traits;

// locale_type getloc()const;

#include <regex>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

int main(int, char**)
{
    {
        std::regex_traits<char> t1;
        assert(t1.getloc().name() == "C");
        std::regex_traits<wchar_t> t2;
        assert(t2.getloc().name() == "C");
    }
    {
        std::locale::global(std::locale(LOCALE_en_US_UTF_8));
        std::regex_traits<char> t1;
        assert(t1.getloc().name() == LOCALE_en_US_UTF_8);
        std::regex_traits<wchar_t> t2;
        assert(t2.getloc().name() == LOCALE_en_US_UTF_8);
    }

  return 0;
}
