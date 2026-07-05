//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.fr_FR.UTF-8

// <streambuf>

// template <class charT, class traits = char_traits<charT> >
// class basic_streambuf;

// locale pubimbue(const locale& loc);
// locale getloc() const;

#include <streambuf>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

template <class CharT>
struct test
    : public std::basic_streambuf<CharT>
{
    test() {}

    void imbue(const std::locale&)
    {
        assert(this->getloc().name() == LOCALE_en_US_UTF_8);
    }
};

int main(int, char**)
{
    {
        test<char> t;
        assert(t.getloc().name() == "C");
    }
    std::locale::global(std::locale(LOCALE_en_US_UTF_8));
    {
        test<char> t;
        assert(t.getloc().name() == LOCALE_en_US_UTF_8);
        assert(t.pubimbue(std::locale(LOCALE_fr_FR_UTF_8)).name() ==
               LOCALE_en_US_UTF_8);
        assert(t.getloc().name() == LOCALE_fr_FR_UTF_8);
    }

  return 0;
}
