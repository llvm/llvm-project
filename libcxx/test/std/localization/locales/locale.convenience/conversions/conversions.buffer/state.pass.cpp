//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// wbuffer_convert<Codecvt, Elem, Tr>

// state_type state() const;

// XFAIL: libcpp-has-no-wide-characters

#include <locale>
#include <codecvt>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::wbuffer_convert<std::codecvt_utf8<wchar_t> > B;
    {
        B b;
        std::mbstate_t s = b.state();
        ((void)s);
    }

  return 0;
}
