//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: locale.en_US.UTF-8

// <locale>

// basic_string<char> name() const;

#include <locale>
#include <cassert>

#include "platform_support.h" // locale name macros

int main(int, char**)
{
    {
        std::locale loc;
        assert(loc.name() == "C");
    }
    {
        std::locale loc(LOCALE_en_US_UTF_8);
        assert(loc.name() == LOCALE_en_US_UTF_8);
    }

  return 0;
}
