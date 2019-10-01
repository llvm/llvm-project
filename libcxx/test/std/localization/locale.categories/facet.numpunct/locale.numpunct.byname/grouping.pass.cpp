//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NetBSD does not support LC_NUMERIC at the moment
// XFAIL: netbsd

// REQUIRES: locale.en_US.UTF-8
// REQUIRES: locale.fr_FR.UTF-8

// <locale>

// template <class charT> class numpunct_byname;

// string grouping() const;

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
            assert(np.grouping() == "");
        }
        {
            typedef wchar_t C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.grouping() == "");
        }
    }
    {
        std::locale l(LOCALE_en_US_UTF_8);
        {
            typedef char C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.grouping() == "\3\3");
        }
        {
            typedef wchar_t C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.grouping() == "\3\3");
        }
    }
    {
        std::locale l(LOCALE_fr_FR_UTF_8);
#if defined(TEST_HAS_GLIBC)
        const char* const group = "\3";
#else
        const char* const group = "\x7f";
#endif
        {
            typedef char C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.grouping() ==  group);
        }
        {
            typedef wchar_t C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.grouping() == group);
        }
    }

  return 0;
}
