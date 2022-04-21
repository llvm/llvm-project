//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NetBSD does not support LC_MONETARY at the moment
// XFAIL: netbsd
// XFAIL: LIBCXX-AIX-FIXME

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ru_RU.UTF-8
// REQUIRES: locale.zh_CN.UTF-8

// <locale>

// class moneypunct_byname<charT, International>

// string_type curr_symbol() const;

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
        assert(f.curr_symbol() == std::string());
    }
    {
        Fnt f("C", 1);
        assert(f.curr_symbol() == std::string());
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f("C", 1);
        assert(f.curr_symbol() == std::wstring());
    }
    {
        Fwt f("C", 1);
        assert(f.curr_symbol() == std::wstring());
    }
#endif

#ifdef _WIN32
    std::string curr_space = "";
#else
    std::string curr_space = " ";
#endif
    {
        Fnf f(LOCALE_en_US_UTF_8, 1);
        assert(f.curr_symbol() == "$");
    }
    {
        Fnt f(LOCALE_en_US_UTF_8, 1);
        assert(f.curr_symbol() == "USD" + curr_space);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
#ifdef _WIN32
    std::wstring w_curr_space = L"";
#else
    std::wstring w_curr_space = L" ";
#endif
    {
        Fwf f(LOCALE_en_US_UTF_8, 1);
        assert(f.curr_symbol() == L"$");
    }
    {
        Fwt f(LOCALE_en_US_UTF_8, 1);
        assert(f.curr_symbol() == L"USD" + w_curr_space);
    }
#endif

    {
        Fnf f(LOCALE_fr_FR_UTF_8, 1);
#ifdef __APPLE__
        assert(f.curr_symbol() == " Eu");
#else
        assert(f.curr_symbol() == " \u20ac");
#endif
    }
    {
        Fnt f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.curr_symbol() == " EUR");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f(LOCALE_fr_FR_UTF_8, 1);
#ifdef __APPLE__
        assert(f.curr_symbol() == L" Eu");
#else
        assert(f.curr_symbol() == L" \u20ac");
#endif
    }
    {
        Fwt f(LOCALE_fr_FR_UTF_8, 1);
        assert(f.curr_symbol() == L" EUR");
    }
#endif

    {
        Fnf f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.curr_symbol() == " " + static_cast<std::string>(LocaleHelpers::currency_symbol_ru_RU()));
    }
    {
        Fnt f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.curr_symbol() == " RUB");
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.curr_symbol() == L" " + static_cast<std::wstring>(LocaleHelpers::currency_symbol_ru_RU()));
    }

    {
        Fwt f(LOCALE_ru_RU_UTF_8, 1);
        assert(f.curr_symbol() == L" RUB");
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

    {
        Fnf f(LOCALE_zh_CN_UTF_8, 1);
#ifdef _WIN32
        assert(f.curr_symbol() == "\xC2\xA5"); // \u00A5
#else
        assert(f.curr_symbol() == "\xEF\xBF\xA5"); // \uFFE5
#endif
    }
    {
        Fnt f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.curr_symbol() == "CNY" + curr_space);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f(LOCALE_zh_CN_UTF_8, 1);
#ifdef _WIN32
        assert(f.curr_symbol() == L"\u00A5");
#else
        assert(f.curr_symbol() == L"\uFFE5");
#endif
    }
    {
        Fwt f(LOCALE_zh_CN_UTF_8, 1);
        assert(f.curr_symbol() == L"CNY" + w_curr_space);
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
