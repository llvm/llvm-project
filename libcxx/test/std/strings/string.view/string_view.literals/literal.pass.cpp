//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Note: libc++ supports string_view before C++17, but literals were introduced in C++14
// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: !stdlib=libc++ && c++14

#include <string_view>
#include <cassert>

#include "test_macros.h"

#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
    typedef std::u8string_view u8string_view;
#else
    typedef std::string_view   u8string_view;
#endif

int main(int, char**) {
    {
        using namespace std::literals::string_view_literals;

        ASSERT_SAME_TYPE(decltype(  "Hi"sv), std::string_view);
        ASSERT_SAME_TYPE(decltype(u8"Hi"sv), u8string_view);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        ASSERT_SAME_TYPE(decltype( L"Hi"sv), std::wstring_view);
#endif
        ASSERT_SAME_TYPE(decltype( u"Hi"sv), std::u16string_view);
        ASSERT_SAME_TYPE(decltype( U"Hi"sv), std::u32string_view);

        std::string_view foo;
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        std::wstring_view Lfoo;
#endif
        u8string_view u8foo;
        std::u16string_view ufoo;
        std::u32string_view Ufoo;

        foo  =    ""sv;     assert(  foo.size() == 0);
        u8foo = u8""sv;     assert(u8foo.size() == 0);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        Lfoo  =  L""sv;     assert( Lfoo.size() == 0);
#endif
        ufoo  =  u""sv;     assert( ufoo.size() == 0);
        Ufoo  =  U""sv;     assert( Ufoo.size() == 0);

        foo   =   " "sv;    assert(  foo.size() == 1);
        u8foo = u8" "sv;    assert(u8foo.size() == 1);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        Lfoo  =  L" "sv;    assert( Lfoo.size() == 1);
#endif
        ufoo  =  u" "sv;    assert( ufoo.size() == 1);
        Ufoo  =  U" "sv;    assert( Ufoo.size() == 1);

        foo   =   "ABC"sv;  assert(  foo ==   "ABC");   assert(  foo == std::string_view   (  "ABC"));
        u8foo = u8"ABC"sv;  assert(u8foo == u8"ABC");   assert(u8foo == u8string_view      (u8"ABC"));
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        Lfoo  =  L"ABC"sv;  assert( Lfoo ==  L"ABC");   assert( Lfoo == std::wstring_view  ( L"ABC"));
#endif
        ufoo  =  u"ABC"sv;  assert( ufoo ==  u"ABC");   assert( ufoo == std::u16string_view( u"ABC"));
        Ufoo  =  U"ABC"sv;  assert( Ufoo ==  U"ABC");   assert( Ufoo == std::u32string_view( U"ABC"));

        static_assert(  "ABC"sv.size() == 3, "");
        static_assert(u8"ABC"sv.size() == 3, "");
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        static_assert( L"ABC"sv.size() == 3, "");
#endif
        static_assert( u"ABC"sv.size() == 3, "");
        static_assert( U"ABC"sv.size() == 3, "");

        ASSERT_NOEXCEPT(  "ABC"sv);
        ASSERT_NOEXCEPT(u8"ABC"sv);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        ASSERT_NOEXCEPT( L"ABC"sv);
#endif
        ASSERT_NOEXCEPT( u"ABC"sv);
        ASSERT_NOEXCEPT( U"ABC"sv);
    }
    {
        using namespace std::literals;
        std::string_view foo = ""sv;
        assert(foo.length() == 0);
    }
    {
        using namespace std;
        std::string_view foo = ""sv;
        assert(foo.length() == 0);
    }

    return 0;
}
