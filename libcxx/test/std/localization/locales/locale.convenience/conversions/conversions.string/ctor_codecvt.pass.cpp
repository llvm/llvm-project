//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS -D_LIBCPP_ENABLE_CXX26_REMOVED_CODECVT -D_LIBCPP_ENABLE_CXX26_REMOVED_WSTRING_CONVERT

// wstring_convert<Codecvt, Elem, Wide_alloc, Byte_alloc>

// wstring_convert(Codecvt* pcvt = new Codecvt);          // before C++14
// explicit wstring_convert(Codecvt* pcvt = new Codecvt); // before C++20
// wstring_convert() : wstring_convert(new Codecvt) {}    // C++20
// explicit wstring_convert(Codecvt* pcvt);               // C++20

// XFAIL: no-wide-characters

#include <locale>
#include <codecvt>
#include <cassert>

#include "test_macros.h"
#if TEST_STD_VER >= 11
#include "test_convertible.h"
#endif

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

#if TEST_STD_VER >= 11
    {
      typedef std::codecvt_utf8<wchar_t> Codecvt;
      typedef std::wstring_convert<Codecvt> B;
      static_assert(test_convertible<B>(), "");
      static_assert(!test_convertible<B, Codecvt*>(), "");
    }
#endif

    return 0;
}
