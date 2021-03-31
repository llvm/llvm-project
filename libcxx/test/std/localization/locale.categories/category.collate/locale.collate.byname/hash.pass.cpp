//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8

// XFAIL: LIBCXX-WINDOWS-FIXME

// <locale>

// template <class charT> class collate_byname

// long hash(const charT* low, const charT* high) const;

//   This test is not portable

#include <locale>
#include <string>
#include <cassert>

#include "test_macros.h"
#include "platform_support.h" // locale name macros

int main(int, char**)
{
    std::locale l(LOCALE_en_US_UTF_8);
    {
        std::string x1("1234");
        std::string x2("12345");
        const std::collate<char>& f = std::use_facet<std::collate<char> >(l);
        assert(f.hash(x1.data(), x1.data() + x1.size())
            != f.hash(x2.data(), x2.data() + x2.size()));
    }
    {
        std::wstring x1(L"1234");
        std::wstring x2(L"12345");
        const std::collate<wchar_t>& f = std::use_facet<std::collate<wchar_t> >(l);
        assert(f.hash(x1.data(), x1.data() + x1.size())
            != f.hash(x2.data(), x2.data() + x2.size()));
    }

  return 0;
}
