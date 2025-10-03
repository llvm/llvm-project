//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.zh_CN.UTF-8

// <locale>

// Test locale name construction for the following constructors:
// locale(const locale& other, const char* std_name, category cat);
// locale(const locale& other, const string& std_name, category cat);
// locale(const locale& other, const locale& one, category cats);

// This test exercises the fix for locale name construction (http://llvm.org/D119441 aka 643b7bcdb404),
// which isn't in the dylib for some systems.
// XFAIL: using-built-library-before-llvm-17

#include <locale>
#include <cassert>
#include "platform_support.h" // locale name macros

int main(int, char**) {
    std::locale en(LOCALE_en_US_UTF_8);
    std::locale zh(LOCALE_zh_CN_UTF_8);
    std::locale unnamed(std::locale(), new std::ctype<char>);
    {
        std::locale loc(unnamed, en, std::locale::time);
        assert(loc.name() == "*");
    }
    {
        std::locale loc(en, unnamed, std::locale::none);
        assert(loc.name() == "*");
    }
    {
        std::locale loc(en, "", std::locale::none);
        assert(loc.name() == en.name());
    }
    {
        std::locale loc(en, zh, std::locale::none);
        assert(loc.name() == en.name());
    }
    {
        std::locale loc(en, LOCALE_en_US_UTF_8, std::locale::time);
        assert(loc.name() == en.name());
    }
    {
        std::locale loc(en, std::string(LOCALE_en_US_UTF_8), std::locale::collate);
        assert(loc.name() == en.name());
    }
  return 0;
}
