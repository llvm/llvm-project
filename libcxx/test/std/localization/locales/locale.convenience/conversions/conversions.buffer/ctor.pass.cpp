//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// wbuffer_convert<Codecvt, Elem, Tr>

// wbuffer_convert(streambuf* bytebuf = 0, Codecvt* pcvt = new Codecvt,
//                 state_type state = state_type());          // before C++14
// explicit wbuffer_convert(streambuf* bytebuf = nullptr, Codecvt* pcvt = new Codecvt,
//                          state_type state = state_type()); // before C++20
// wbuffer_convert() : wbuffer_convert(nullptr) {} // C++20
// explicit wbuffer_convert(streambuf* bytebuf, Codecvt* pcvt = new Codecvt,
//                          state_type state = state_type()); // C++20

// XFAIL: no-wide-characters

#include <locale>
#include <codecvt>
#include <sstream>
#include <cassert>

#include "test_macros.h"
#include "count_new.h"
#if TEST_STD_VER >= 11
#include "test_convertible.h"
#endif

int main(int, char**)
{
    globalMemCounter.reset();
    typedef std::wbuffer_convert<std::codecvt_utf8<wchar_t> > B;
#if TEST_STD_VER > 11
    static_assert(!std::is_convertible<std::streambuf*, B>::value, "");
    static_assert( std::is_constructible<B, std::streambuf*>::value, "");
#endif
    {
        B b;
        assert(b.rdbuf() == nullptr);
        assert(globalMemCounter.checkOutstandingNewNotEq(0));
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
        std::stringstream s;
        B b(s.rdbuf());
        assert(b.rdbuf() == s.rdbuf());
        assert(globalMemCounter.checkOutstandingNewNotEq(0));
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
        std::stringstream s;
        B b(s.rdbuf(), new std::codecvt_utf8<wchar_t>);
        assert(b.rdbuf() == s.rdbuf());
        assert(globalMemCounter.checkOutstandingNewNotEq(0));
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
        std::stringstream s;
        B b(s.rdbuf(), new std::codecvt_utf8<wchar_t>, std::mbstate_t());
        assert(b.rdbuf() == s.rdbuf());
        assert(globalMemCounter.checkOutstandingNewNotEq(0));
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));

#if TEST_STD_VER >= 11
    {
      static_assert(test_convertible<B>(), "");
      static_assert(!test_convertible<B, std::streambuf*>(), "");
    }
#endif

    return 0;
}
