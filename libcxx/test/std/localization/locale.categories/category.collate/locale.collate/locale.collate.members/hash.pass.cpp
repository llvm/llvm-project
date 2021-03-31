//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <class charT> class collate;

// long hash(const charT* low, const charT* high) const;

//   This test is not portable

// XFAIL: LIBCXX-WINDOWS-FIXME

#include <locale>
#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::locale l = std::locale::classic();
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
