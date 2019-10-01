//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// wstring_convert<Codecvt, Elem, Wide_alloc, Byte_alloc>

// state_type state() const;

#include <locale>
#include <codecvt>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::codecvt_utf8<wchar_t> Codecvt;
    typedef std::wstring_convert<Codecvt> Myconv;
    Myconv myconv;
    std::mbstate_t s = myconv.state();
    ((void)s);

  return 0;
}
