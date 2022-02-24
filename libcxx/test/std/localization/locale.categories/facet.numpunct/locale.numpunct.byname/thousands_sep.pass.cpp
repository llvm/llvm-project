//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NetBSD does not support LC_NUMERIC at the moment
// XFAIL: netbsd

// XFAIL: LIBCXX-AIX-FIXME

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.fr_FR.UTF-8

// <locale>

// template <class charT> class numpunct_byname;

// char_type thousands_sep() const;


#include <locale>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

int main(int, char**)
{
    {
        std::locale l("C");
        {
            typedef char C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.thousands_sep() == ',');
        }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        {
            typedef wchar_t C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.thousands_sep() == L',');
        }
#endif
    }
    {
        std::locale l(LOCALE_en_US_UTF_8);
        {
            typedef char C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.thousands_sep() == ',');
        }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        {
            typedef wchar_t C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.thousands_sep() == L',');
        }
#endif
    }
    {
        // The below tests work around GLIBC's use of U202F as LC_NUMERIC thousands_sep.
        std::locale l(LOCALE_fr_FR_UTF_8);
        {
#if defined(_CS_GNU_LIBC_VERSION) || defined(_WIN32)
            const char sep = ' ';
#else
            const char sep = ',';
#endif
            typedef char C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.thousands_sep() == sep);
        }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        {
#if defined(_CS_GNU_LIBC_VERSION)
            const wchar_t wsep = glibc_version_less_than("2.27") ? L' ' : L'\u202f';
#elif defined(_WIN32)
            const wchar_t wsep = L'\u00A0';
#else
            const wchar_t wsep = L',';
#endif
            typedef wchar_t C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.thousands_sep() == wsep);
        }
#endif // TEST_HAS_NO_WIDE_CHARACTERS
    }

    return 0;
}
