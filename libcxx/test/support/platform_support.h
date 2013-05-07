//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Define a bunch of macros that can be used in the tests instead of
//  implementation defined assumptions:
//   - locale names
//   - floating point number string output

#ifndef PLATFORM_SUPPORT_H
#define PLATFORM_SUPPORT_H

// locale names
#ifdef _WIN32
// WARNING: Windows does not support UTF-8 codepages.
// Locales are "converted" using http://docs.moodle.org/dev/Table_of_locales
#define LOCALE_en_US_UTF_8     "English_United States.1252"
#define LOCALE_cs_CZ_ISO8859_2 "Czech_Czech Republic.1250"
#define LOCALE_fr_FR_UTF_8     "French_France.1252"
#define LOCALE_fr_CA_ISO8859_1 "French_Canada.1252"
#define LOCALE_ru_RU_UTF_8     "Russian_Russia.1251"
#define LOCALE_zh_CN_UTF_8     "Chinese_China.936"
#else
#define LOCALE_en_US_UTF_8     "en_US.UTF-8"
#define LOCALE_fr_FR_UTF_8     "fr_FR.UTF-8"
#ifdef __linux__
#define LOCALE_fr_CA_ISO8859_1 "fr_CA.ISO-8859-1"
#define LOCALE_cs_CZ_ISO8859_2 "cs_CZ.ISO-8859-2"
#else
#define LOCALE_fr_CA_ISO8859_1 "fr_CA.ISO8859-1"
#define LOCALE_cs_CZ_ISO8859_2 "cs_CZ.ISO8859-2"
#endif
#define LOCALE_ru_RU_UTF_8     "ru_RU.UTF-8"
#define LOCALE_zh_CN_UTF_8     "zh_CN.UTF-8"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string>

inline
std::string
get_temp_file_name()
{
#ifdef _WIN32
   char* p = _tempnam( NULL, NULL );
   if (p == nullptr)
       abort();
    std::string s(p);
    free( p );
#else
   std::string s("temp.XXXXXX");
   mktemp(&s[0]);
#endif
   return s;
}

#endif // PLATFORM_SUPPORT_H
