//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class numpunct_byname;

// char_type decimal_point() const;

#include <locale>
#include <cassert>

#include "../../../../platform_support.h" // locale name macros

int main()
{
    {
        std::locale l("C");
        {
            typedef char C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.decimal_point() == '.');
        }
        {
            typedef wchar_t C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.decimal_point() == L'.');
        }
    }
    {
        std::locale l(LOCALE_en_US_UTF_8);
        {
            typedef char C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.decimal_point() == '.');
        }
        {
            typedef wchar_t C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.decimal_point() == L'.');
        }
    }
    {
        std::locale l(LOCALE_fr_FR_UTF_8);
        {
            typedef char C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.decimal_point() == ',');
        }
        {
            typedef wchar_t C;
            const std::numpunct<C>& np = std::use_facet<std::numpunct<C> >(l);
            assert(np.decimal_point() == L',');
        }
    }
}
