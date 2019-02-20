//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// wstring_convert<Codecvt, Elem, Wide_alloc, Byte_alloc>

// wstring_convert(Codecvt* pcvt = new Codecvt);

#include <locale>
#include <codecvt>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef std::codecvt_utf8<wchar_t> Codecvt;
        typedef std::wstring_convert<Codecvt> Myconv;
        Myconv myconv;
        assert(myconv.converted() == 0);
    }
    {
        typedef std::codecvt_utf8<wchar_t> Codecvt;
        typedef std::wstring_convert<Codecvt> Myconv;
        Myconv myconv(new Codecvt);
        assert(myconv.converted() == 0);
#if TEST_STD_VER > 11
        static_assert(!std::is_convertible<Codecvt*, Myconv>::value, "");
        static_assert( std::is_constructible<Myconv, Codecvt*>::value, "");
#endif
    }

  return 0;
}
