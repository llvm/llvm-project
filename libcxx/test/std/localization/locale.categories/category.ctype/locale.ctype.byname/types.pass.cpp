//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8

// <locale>

// template <class CharT>
// class ctype_byname
//     : public ctype<CharT>
// {
// public:
//     explicit ctype_byname(const char*, size_t = 0);
//     explicit ctype_byname(const string&, size_t = 0);
//
// protected:
//     ~ctype_byname();
// };

#include <locale>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

int main(int, char**)
{
    {
        std::locale l(LOCALE_en_US_UTF_8);
        {
            assert(std::has_facet<std::ctype_byname<char> >(l));
            assert(&std::use_facet<std::ctype<char> >(l)
                == &std::use_facet<std::ctype_byname<char> >(l));
        }
        {
            assert(std::has_facet<std::ctype_byname<wchar_t> >(l));
            assert(&std::use_facet<std::ctype<wchar_t> >(l)
                == &std::use_facet<std::ctype_byname<wchar_t> >(l));
        }
    }
    {
        std::locale l("C");
        {
            assert(std::has_facet<std::ctype_byname<char> >(l));
            assert(&std::use_facet<std::ctype<char> >(l)
                == &std::use_facet<std::ctype_byname<char> >(l));
        }
        {
            assert(std::has_facet<std::ctype_byname<wchar_t> >(l));
            assert(&std::use_facet<std::ctype<wchar_t> >(l)
                == &std::use_facet<std::ctype_byname<wchar_t> >(l));
        }
    }

  return 0;
}
