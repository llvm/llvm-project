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

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ru_RU.UTF-8
// REQUIRES: locale.zh_CN.UTF-8

// <locale>

// class moneypunct_byname<charT, International>

// pattern neg_format() const;

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

void assert_symbol_sign_none_value(std::money_base::pattern p)
{
    assert(p.field[0] == std::money_base::symbol);
    assert(p.field[1] == std::money_base::sign);
    assert(p.field[2] == std::money_base::none);
    assert(p.field[3] == std::money_base::value);
}

void assert_sign_symbol_none_value(std::money_base::pattern p)
{
    assert(p.field[0] == std::money_base::sign);
    assert(p.field[1] == std::money_base::symbol);
    assert(p.field[2] == std::money_base::none);
    assert(p.field[3] == std::money_base::value);
}

void assert_value_none_symbol_sign(std::money_base::pattern p)
{
    assert(p.field[0] == std::money_base::value);
    assert(p.field[1] == std::money_base::none);
    assert(p.field[2] == std::money_base::symbol);
    assert(p.field[3] == std::money_base::sign);
}

void assert_sign_value_none_symbol(std::money_base::pattern p)
{
    assert(p.field[0] == std::money_base::sign);
    assert(p.field[1] == std::money_base::value);
    assert(p.field[2] == std::money_base::none);
    assert(p.field[3] == std::money_base::symbol);
}

int main(int, char**)
{
    {
        Fnf f("C", 1);
        std::money_base::pattern p = f.neg_format();
        assert_symbol_sign_none_value(p);
    }
    {
        Fnt f("C", 1);
        std::money_base::pattern p = f.neg_format();
        assert_symbol_sign_none_value(p);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f("C", 1);
        std::money_base::pattern p = f.neg_format();
        assert_symbol_sign_none_value(p);
    }
    {
        Fwt f("C", 1);
        std::money_base::pattern p = f.neg_format();
        assert_symbol_sign_none_value(p);
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

    {
        Fnf f(LOCALE_en_US_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
        assert_sign_symbol_none_value(p);
    }
    {
        Fnt f(LOCALE_en_US_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
        assert_sign_symbol_none_value(p);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f(LOCALE_en_US_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
        assert_sign_symbol_none_value(p);
    }
    {
        Fwt f(LOCALE_en_US_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
        assert_sign_symbol_none_value(p);
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

    {
        Fnf f(LOCALE_fr_FR_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
#ifdef __APPLE__
        assert_value_none_symbol_sign(p);
#else
        assert_sign_value_none_symbol(p);
#endif
    }
    {
        Fnt f(LOCALE_fr_FR_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
#ifdef __APPLE__
        assert_value_none_symbol_sign(p);
#else
        assert_sign_value_none_symbol(p);
#endif
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f(LOCALE_fr_FR_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
#ifdef __APPLE__
        assert_value_none_symbol_sign(p);
#else
        assert_sign_value_none_symbol(p);
#endif
    }
    {
        Fwt f(LOCALE_fr_FR_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
#ifdef __APPLE__
        assert_value_none_symbol_sign(p);
#else
        assert_sign_value_none_symbol(p);
#endif
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

    {
        Fnf f(LOCALE_ru_RU_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
        assert_sign_value_none_symbol(p);
    }
    {
        Fnt f(LOCALE_ru_RU_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
        assert_sign_value_none_symbol(p);
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f(LOCALE_ru_RU_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
        assert_sign_value_none_symbol(p);
    }
    {
        Fwt f(LOCALE_ru_RU_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
        assert_sign_value_none_symbol(p);
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

    {
        Fnf f(LOCALE_zh_CN_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
#ifdef _AIX
        assert_sign_symbol_none_value(p);
#else
        assert_symbol_sign_none_value(p);
#endif
    }
    {
        Fnt f(LOCALE_zh_CN_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
#if defined(_WIN32) || defined(__APPLE__)
        assert_symbol_sign_none_value(p);
#else
        assert_sign_symbol_none_value(p);
#endif
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        Fwf f(LOCALE_zh_CN_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
#ifdef _AIX
        assert_sign_symbol_none_value(p);
#else
        assert_symbol_sign_none_value(p);
#endif
    }
    {
        Fwt f(LOCALE_zh_CN_UTF_8, 1);
        std::money_base::pattern p = f.neg_format();
#if defined(_WIN32) || defined(__APPLE__)
        assert_symbol_sign_none_value(p);
#else
        assert_sign_symbol_none_value(p);
#endif
    }
#endif // TEST_HAS_NO_WIDE_CHARACTERS

  return 0;
}
