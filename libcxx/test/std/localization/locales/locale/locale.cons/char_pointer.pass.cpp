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
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.15
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.14
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.13
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.12
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.11
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.10
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.9

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


void check(const std::locale& loc)
{
    assert(std::has_facet<std::collate<char> >(loc));
    assert(std::has_facet<std::collate<wchar_t> >(loc));

    assert(std::has_facet<std::ctype<char> >(loc));
    assert(std::has_facet<std::ctype<wchar_t> >(loc));
    assert((std::has_facet<std::codecvt<char, char, std::mbstate_t> >(loc)));
    assert((std::has_facet<std::codecvt<char16_t, char, std::mbstate_t> >(loc)));
    assert((std::has_facet<std::codecvt<char32_t, char, std::mbstate_t> >(loc)));
#if TEST_STD_VER > 17
    assert((std::has_facet<std::codecvt<char16_t, char8_t, std::mbstate_t> >(loc)));
    assert((std::has_facet<std::codecvt<char32_t, char8_t, std::mbstate_t> >(loc)));
#endif
    assert((std::has_facet<std::codecvt<wchar_t, char, std::mbstate_t> >(loc)));

    assert((std::has_facet<std::moneypunct<char> >(loc)));
    assert((std::has_facet<std::moneypunct<wchar_t> >(loc)));
    assert((std::has_facet<std::money_get<char> >(loc)));
    assert((std::has_facet<std::money_get<wchar_t> >(loc)));
    assert((std::has_facet<std::money_put<char> >(loc)));
    assert((std::has_facet<std::money_put<wchar_t> >(loc)));

    assert((std::has_facet<std::numpunct<char> >(loc)));
    assert((std::has_facet<std::numpunct<wchar_t> >(loc)));
    assert((std::has_facet<std::num_get<char> >(loc)));
    assert((std::has_facet<std::num_get<wchar_t> >(loc)));
    assert((std::has_facet<std::num_put<char> >(loc)));
    assert((std::has_facet<std::num_put<wchar_t> >(loc)));

    assert((std::has_facet<std::time_get<char> >(loc)));
    assert((std::has_facet<std::time_get<wchar_t> >(loc)));
    assert((std::has_facet<std::time_put<char> >(loc)));
    assert((std::has_facet<std::time_put<wchar_t> >(loc)));

    assert((std::has_facet<std::messages<char> >(loc)));
    assert((std::has_facet<std::messages<wchar_t> >(loc)));
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
