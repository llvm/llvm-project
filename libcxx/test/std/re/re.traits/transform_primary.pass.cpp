//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NetBSD does not support LC_COLLATE at the moment
// XFAIL: netbsd

// XFAIL: LIBCXX-AIX-FIXME

// REQUIRES: locale.cs_CZ.ISO8859-2

// <regex>

// template <class charT> struct regex_traits;

// template <class ForwardIterator>
//   string_type
//   transform_primary(ForwardIterator first, ForwardIterator last) const;

#include <regex>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "platform_support.h" // locale name macros

int main(int, char**)
{
    {
        std::regex_traits<char> t;
        const char A[] = "A";
        const char AA[]      = "AA";
        const char B[]       = "B";
        const char BB[]      = "BB";
        const char Aacute[]  = "\xC1";
        const char AAacute[] = "\xC1\xC1";
        typedef forward_iterator<const char*> F;
        assert(t.transform_primary(F(A), F(A+1)) !=
               t.transform_primary(F(Aacute), F(Aacute+1)));
        assert(t.transform_primary(F(B), F(B + 1)) != t.transform_primary(F(Aacute), F(Aacute + 1)));
        assert(t.transform_primary(F(A), F(A + 1)) != t.transform_primary(F(AAacute), F(AAacute + 2)));
        assert(t.transform_primary(F(AA), F(AA + 2)) != t.transform_primary(F(AAacute), F(AAacute + 2)));
        assert(t.transform_primary(F(BB), F(BB + 2)) != t.transform_primary(F(AAacute), F(AAacute + 2)));
        assert(t.transform_primary(F(AA), F(AA + 2)) != t.transform_primary(F(BB), F(BB + 2)));
        t.imbue(std::locale(LOCALE_cs_CZ_ISO8859_2));
        assert(t.transform_primary(F(A), F(A+1)) ==
               t.transform_primary(F(Aacute), F(Aacute+1)));
        assert(t.transform_primary(F(B), F(B + 1)) != t.transform_primary(F(Aacute), F(Aacute + 1)));
        assert(t.transform_primary(F(A), F(A + 1)) != t.transform_primary(F(AAacute), F(AAacute + 2)));
        assert(t.transform_primary(F(AA), F(AA + 2)) == t.transform_primary(F(AAacute), F(AAacute + 2)));
        assert(t.transform_primary(F(BB), F(BB + 2)) != t.transform_primary(F(AAacute), F(AAacute + 2)));
        assert(t.transform_primary(F(AA), F(AA + 2)) != t.transform_primary(F(BB), F(BB + 2)));
    }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    {
        std::regex_traits<wchar_t> t;
        const wchar_t A[] = L"A";
        const wchar_t AA[]      = L"AA";
        const wchar_t B[]       = L"B";
        const wchar_t BB[]      = L"BB";
        const wchar_t Aacute[] = L"\xC1";
        const wchar_t AAacute[] = L"\xC1\xC1";
        typedef forward_iterator<const wchar_t*> F;
        assert(t.transform_primary(F(A), F(A+1)) !=
               t.transform_primary(F(Aacute), F(Aacute+1)));
        assert(t.transform_primary(F(B), F(B + 1)) != t.transform_primary(F(Aacute), F(Aacute + 1)));
        assert(t.transform_primary(F(A), F(A + 1)) != t.transform_primary(F(AAacute), F(AAacute + 2)));
        assert(t.transform_primary(F(AA), F(AA + 2)) != t.transform_primary(F(AAacute), F(AAacute + 2)));
        assert(t.transform_primary(F(BB), F(BB + 2)) != t.transform_primary(F(AAacute), F(AAacute + 2)));
        assert(t.transform_primary(F(AA), F(AA + 2)) != t.transform_primary(F(BB), F(BB + 2)));
        t.imbue(std::locale(LOCALE_cs_CZ_ISO8859_2));
        assert(t.transform_primary(F(A), F(A+1)) ==
               t.transform_primary(F(Aacute), F(Aacute+1)));
        assert(t.transform_primary(F(B), F(B + 1)) != t.transform_primary(F(Aacute), F(Aacute + 1)));
        assert(t.transform_primary(F(A), F(A + 1)) != t.transform_primary(F(AAacute), F(AAacute + 2)));
        assert(t.transform_primary(F(AA), F(AA + 2)) == t.transform_primary(F(AAacute), F(AAacute + 2)));
        assert(t.transform_primary(F(BB), F(BB + 2)) != t.transform_primary(F(AAacute), F(AAacute + 2)));
        assert(t.transform_primary(F(AA), F(AA + 2)) != t.transform_primary(F(BB), F(BB + 2)));
    }
#endif

  return 0;
}
