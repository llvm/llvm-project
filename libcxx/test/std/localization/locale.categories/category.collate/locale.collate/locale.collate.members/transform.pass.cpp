//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class collate;

// string_type transform(const charT* low, const charT* high) const;

#include <locale>
#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::locale l = std::locale::classic();
    {
        std::string x("1234");
        const std::collate<char>& f = std::use_facet<std::collate<char> >(l);
        assert(f.transform(x.data(), x.data() + x.size()) == x);
    }
    {
        std::wstring x(L"1234");
        const std::collate<wchar_t>& f = std::use_facet<std::collate<wchar_t> >(l);
        assert(f.transform(x.data(), x.data() + x.size()) == x);
    }

  return 0;
}
