//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// wstring_convert<Codecvt, Elem, Wide_alloc, Byte_alloc>

// wstring_convert(Codecvt* pcvt, state_type state);

#include <locale>
#include <codecvt>
#include <cassert>

int main(int, char**)
{
    {
        typedef std::codecvt_utf8<wchar_t> Codecvt;
        typedef std::wstring_convert<Codecvt> Myconv;
        Myconv myconv(new Codecvt, std::mbstate_t());
        assert(myconv.converted() == 0);
    }

  return 0;
}
