//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NetBSD does not support LC_MONETARY at the moment
// XFAIL: netbsd

// XFAIL: LIBCXX-FREEBSD-FIXME

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ru_RU.UTF-8
// REQUIRES: locale.zh_CN.UTF-8

// ADDITIONAL_COMPILE_FLAGS: -DFR_MON_THOU_SEP=%{LOCALE_CONV_FR_FR_UTF_8_MON_THOUSANDS_SEP}
// ADDITIONAL_COMPILE_FLAGS: -DRU_MON_THOU_SEP=%{LOCALE_CONV_RU_RU_UTF_8_MON_THOUSANDS_SEP}

// <locale>

// class moneypunct_byname<charT, International>

// charT thousands_sep() const;

#include <locale>
#include <limits>
#include <cassert>

#include "test_macros.h"
#include "locale_helpers.h"
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
    {
        Fnf f("C", 1);
        assert(f.thousands_sep() == std::numeric_limits<char>::max());
    }
    {
        Fnt f("C", 1);
        assert(f.thousands_sep() == std::numeric_limits<char>::max());
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f("C", 1);
        assert(f.thousands_sep() == std::numeric_limits<wchar_t>::max());
    }
    {
        Fwt f("C", 1);
        assert(f.thousands_sep() == std::numeric_limits<wchar_t>::max());
    }
#endif

    {
        Fnf f(LOCALE_en_US_UTF_8, 1);
        assert(f.thousands_sep() == ',');
    }
    {
        Fnt f(LOCALE_en_US_UTF_8, 1);
        assert(f.thousands_sep() == ',');
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f(LOCALE_en_US_UTF_8, 1);
        assert(f.thousands_sep() == L',');
    }
    {
        Fwt f(LOCALE_en_US_UTF_8, 1);
        assert(f.thousands_sep() == L',');
    }
#endif
    {
        Fnf f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.thousands_sep() == ' ');
    }
    {
        Fnt f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.thousands_sep() == ' ');
    }

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    const wchar_t fr_sep = LocaleHelpers::mon_thousands_sep_or_default(FR_MON_THOU_SEP);

    {
        Fwf f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.thousands_sep() == fr_sep);
    }
    {
        Fwt f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.thousands_sep() == fr_sep);
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS
    const char sep = ' ';
    {
        Fnf f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.thousands_sep() == sep);
    }
    {
        Fnt f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.thousands_sep() == sep);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    const wchar_t wsep = LocaleHelpers::mon_thousands_sep_or_default(RU_MON_THOU_SEP);

    {
        Fwf f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.thousands_sep() == wsep);
    }
    {
        Fwt f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.thousands_sep() == wsep);
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

    {
        Fnf f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.thousands_sep() == ',');
    }
    {
        Fnt f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.thousands_sep() == ',');
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.thousands_sep() == L',');
    }
    {
        Fwt f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.thousands_sep() == L',');
    }
#endif

  return 0;
}
