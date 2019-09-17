//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// template <> class ctype<char>

// ~ctype();

#include <locale>
#include <cassert>

#include "count_new.h"

#include "test_macros.h"

int main(int, char**)
{
    {
        std::locale l(std::locale::classic(), new std::ctype<char>);
        assert(globalMemCounter.checkDeleteArrayCalledEq(0));
    }
    assert(globalMemCounter.checkDeleteArrayCalledEq(0));
    {
        std::ctype<char>::mask table[256];
        std::locale l(std::locale::classic(), new std::ctype<char>(table));
        assert(globalMemCounter.checkDeleteArrayCalledEq(0));
    }
    assert(globalMemCounter.checkDeleteArrayCalledEq(0));
    {
        std::locale l(std::locale::classic(),
            new std::ctype<char>(new std::ctype<char>::mask[256], true));
        assert(globalMemCounter.checkDeleteArrayCalledEq(0));
    }
    assert(globalMemCounter.checkDeleteArrayCalledEq(1));

  return 0;
}
