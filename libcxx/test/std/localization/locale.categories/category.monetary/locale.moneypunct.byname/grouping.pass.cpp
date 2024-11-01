//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// XFAIL: darwin
//
// NetBSD does not support LC_MONETARY at the moment
// XFAIL: netbsd

// XFAIL: LIBCXX-FREEBSD-FIXME

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ru_RU.UTF-8
// REQUIRES: locale.zh_CN.UTF-8

// <locale>

// class moneypunct_byname<charT, International>

// string grouping() const;

#include <locale>
#include <limits>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

class Fnf
    : public std::moneypunct_byname<char, false>
{
public:
    explicit Fnf(const std::string& nm, std::size_t refs = 0)
        : std::moneypunct_byname<char, false>(nm, refs) {}
};

class Fnt
    : public std::moneypunct_byname<char, true>
{
public:
    explicit Fnt(const std::string& nm, std::size_t refs = 0)
        : std::moneypunct_byname<char, true>(nm, refs) {}
};

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
class Fwf
    : public std::moneypunct_byname<wchar_t, false>
{
public:
    explicit Fwf(const std::string& nm, std::size_t refs = 0)
        : std::moneypunct_byname<wchar_t, false>(nm, refs) {}
};

class Fwt
    : public std::moneypunct_byname<wchar_t, true>
{
public:
    explicit Fwt(const std::string& nm, std::size_t refs = 0)
        : std::moneypunct_byname<wchar_t, true>(nm, refs) {}
};
#endif // TEST_HAS_NO_WIDE_CHARACTERS

int main(int, char**)
{
    // Monetary grouping strings may be terminated with 0 or CHAR_MAX, defining
    // how the grouping is repeated.
    std::string s = std::string(1, CHAR_MAX);
    {
        Fnf f("C", 1);
        assert(f.grouping() == s || f.grouping() == "");
    }
    {
        Fnt f("C", 1);
        assert(f.grouping() == s || f.grouping() == "");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f("C", 1);
        assert(f.grouping() == s || f.grouping() == "");
    }
    {
        Fwt f("C", 1);
        assert(f.grouping() == s || f.grouping() == "");
    }
#endif

#if defined( _WIN32) || defined(_AIX)
    std::string us_grouping = "\3";
#else
    std::string us_grouping = "\3\3";
#endif
    {
        Fnf f(LOCALE_en_US_UTF_8, 1);
        assert(f.grouping() == us_grouping);
    }
    {
        Fnt f(LOCALE_en_US_UTF_8, 1);
        assert(f.grouping() == us_grouping);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f(LOCALE_en_US_UTF_8, 1);
        assert(f.grouping() == us_grouping);
    }
    {
        Fwt f(LOCALE_en_US_UTF_8, 1);
        assert(f.grouping() == us_grouping);
    }
#endif

    {
        Fnf f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.grouping() == "\3");
    }
    {
        Fnt f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.grouping() == "\3");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.grouping() == "\3");
    }
    {
        Fwt f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.grouping() == "\3");
    }
#endif

#if defined( _WIN32) || defined(_AIX)
    std::string ru_grouping = "\3";
#else
    std::string ru_grouping = "\3\3";
#endif
    {
        Fnf f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.grouping() == ru_grouping);
    }
    {
        Fnt f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.grouping() == ru_grouping);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.grouping() == ru_grouping);
    }
    {
        Fwt f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.grouping() == ru_grouping);
    }
#endif

    {
        Fnf f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.grouping() == "\3");
    }
    {
        Fnt f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.grouping() == "\3");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.grouping() == "\3");
    }
    {
        Fwt f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.grouping() == "\3");
    }
#endif

  return 0;
}
