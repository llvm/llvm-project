//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class ctype<char>

// const mask* table() const throw();

#include <locale>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::ctype<char> F;
    {
        std::locale l(std::locale::classic(), new std::ctype<char>);
        const F& f = std::use_facet<F>(l);
        assert(f.table() == f.classic_table());
    }
    {
        std::ctype<char>::mask table[256];
        std::locale l(std::locale::classic(), new std::ctype<char>(table));
        const F& f = std::use_facet<F>(l);
        assert(f.table() == table);
    }

  return 0;
}
