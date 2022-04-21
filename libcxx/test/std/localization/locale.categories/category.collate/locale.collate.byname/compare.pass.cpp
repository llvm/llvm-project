//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8

// <locale>

// template <class charT> class collate_byname

// int compare(const charT* low1, const charT* high1,
//             const charT* low2, const charT* high2) const;

//  I'm currently unable to confirm that collation based on named locales
//     has any difference from "C" collation.  But I do believe I'm picking
//     up the OS's collation files.

// https://llvm.org/PR41018
// XFAIL: windows-dll && msvc

#include <locale>
#include <string>
#include <cassert>

#include <stdio.h>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

#define ASSERT_COMPARE(type, str1, str2, expected) \
    do { \
        type s1(str1); \
        type s2(str2); \
        assert(f.compare(s1.data(), s1.data() + s1.size(), \
                         s2.data(), s2.data() + s2.size()) == (expected)); \
    } while (0)

int main(int, char**)
{
    {
        std::locale l(LOCALE_en_US_UTF_8);
        {
            const std::collate<char>& f = std::use_facet<std::collate<char> >(l);

            ASSERT_COMPARE(std::string, "aaa", "bbb", -1);
            ASSERT_COMPARE(std::string, "AAA", "BBB", -1);
            ASSERT_COMPARE(std::string, "bbb", "aaa", 1);
            ASSERT_COMPARE(std::string, "ccc", "ccc", 0);

#if defined(__APPLE__)
            // Apple's default collation is case-sensitive
            ASSERT_COMPARE(std::string, "aaaaaaA", "BaaaaaA", 1);
#else
            // Glibc, Windows, and FreeBSD's default collation is case-insensitive
            ASSERT_COMPARE(std::string, "aaaaaaA", "BaaaaaA", -1);
#endif
        }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        {
            const std::collate<wchar_t>& f = std::use_facet<std::collate<wchar_t> >(l);

            ASSERT_COMPARE(std::wstring, L"aaa", L"bbb", -1);
            ASSERT_COMPARE(std::wstring, L"AAA", L"BBB", -1);
            ASSERT_COMPARE(std::wstring, L"bbb", L"aaa", 1);
            ASSERT_COMPARE(std::wstring, L"ccc", L"ccc", 0);
#if defined(__APPLE__)
            // Apple's default collation is case-sensitive
            ASSERT_COMPARE(std::wstring, L"aaaaaaA", L"BaaaaaA", 1);
#else
            // Glibc, Windows, and FreeBSD's default collation is case-insensitive
            ASSERT_COMPARE(std::wstring, L"aaaaaaA", L"BaaaaaA", -1);
#endif
        }
#endif
    }
    {
        std::locale l("C");
        {
            const std::collate<char>& f = std::use_facet<std::collate<char> >(l);
            ASSERT_COMPARE(std::string, "aaa", "bbb", -1);
            ASSERT_COMPARE(std::string, "AAA", "BBB", -1);
            ASSERT_COMPARE(std::string, "bbb", "aaa", 1);
            ASSERT_COMPARE(std::string, "ccc", "ccc", 0);

            // In the C locale, these are collated lexicographically.
            ASSERT_COMPARE(std::string, "aaaaaaA", "BaaaaaA", 1);
        }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        {
            const std::collate<wchar_t>& f = std::use_facet<std::collate<wchar_t> >(l);

            ASSERT_COMPARE(std::wstring, L"aaa", L"bbb", -1);
            ASSERT_COMPARE(std::wstring, L"AAA", L"BBB", -1);
            ASSERT_COMPARE(std::wstring, L"bbb", L"aaa", 1);
            ASSERT_COMPARE(std::wstring, L"ccc", L"ccc", 0);

            // In the C locale, these are collated lexicographically.
            ASSERT_COMPARE(std::wstring, L"aaaaaaA", L"BaaaaaA", 1);
        }
#endif
    }

  return 0;
}
