//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NetBSD does not support most of LC_* at the moment
// XFAIL: netbsd

// REQUIRES: locale.ru_RU.UTF-8
// REQUIRES: locale.zh_CN.UTF-8

// This test relies on P0482 being fixed, which isn't in
// older Apple dylibs
//
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14|15}}

// This test runs in C++20, but we have deprecated codecvt<char(16|32), char, mbstate_t> in C++20.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <locale>

// explicit locale(const char* std_name);

#include <locale>
#include <new>
#include <cassert>

#include "count_new.h"
#include "platform_support.h" // locale name macros

#include "test_macros.h"

template<class CharT>
void check_for(const std::locale& loc)
{
    assert(std::has_facet<std::collate<CharT> >(loc));

    assert(std::has_facet<std::ctype<CharT> >(loc));

    assert((std::has_facet<std::codecvt<CharT, char, std::mbstate_t> >(loc)));

    assert(std::has_facet<std::moneypunct<CharT> >(loc));
    assert(std::has_facet<std::money_get<CharT> >(loc));
    assert(std::has_facet<std::money_put<CharT> >(loc));

    assert(std::has_facet<std::numpunct<CharT> >(loc));
    assert(std::has_facet<std::num_get<CharT> >(loc));
    assert(std::has_facet<std::num_put<CharT> >(loc));

    assert(std::has_facet<std::time_get<CharT> >(loc));
    assert(std::has_facet<std::time_put<CharT> >(loc));

    assert(std::has_facet<std::messages<CharT> >(loc));
}

void check(const std::locale& loc)
{
    check_for<char>(loc);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    check_for<wchar_t>(loc);
#endif

    assert((std::has_facet<std::codecvt<char16_t, char, std::mbstate_t> >(loc)));
    assert((std::has_facet<std::codecvt<char32_t, char, std::mbstate_t> >(loc)));
#if TEST_STD_VER > 17
    assert((std::has_facet<std::codecvt<char16_t, char8_t, std::mbstate_t> >(loc)));
    assert((std::has_facet<std::codecvt<char32_t, char8_t, std::mbstate_t> >(loc)));
#endif
}

int main(int, char**)
{
    {
        std::locale loc(LOCALE_ru_RU_UTF_8);
        check(loc);
        std::locale loc2(LOCALE_ru_RU_UTF_8);
        check(loc2);
        assert(loc == loc2);
        std::locale loc3(LOCALE_zh_CN_UTF_8);
        check(loc3);
        assert(!(loc == loc3));
        assert(loc != loc3);
#ifndef TEST_HAS_NO_EXCEPTIONS
        try
        {
            std::locale((const char*)0);
            assert(false);
        }
        catch (std::runtime_error&)
        {
        }
        try
        {
            std::locale("spazbot");
            assert(false);
        }
        catch (std::runtime_error&)
        {
        }
#endif
        std::locale ok("");
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));

  return 0;
}
