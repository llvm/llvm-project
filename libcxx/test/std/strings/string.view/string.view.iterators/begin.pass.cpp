//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !stdlib=libc++ && (c++03 || c++11 || c++14)

// <string_view>

// constexpr const_iterator begin() const;

#include <string_view>
#include <cassert>

#include "test_macros.h"

template <class S>
void
test(S s)
{
    const S& cs = s;
    typename S::iterator b = s.begin();
    typename S::const_iterator cb1 = cs.begin();
    typename S::const_iterator cb2 = s.cbegin();
    if (!s.empty())
    {
        assert(   *b ==  s[0]);
        assert(  &*b == &s[0]);
        assert( *cb1 ==  s[0]);
        assert(&*cb1 == &s[0]);
        assert( *cb2 ==  s[0]);
        assert(&*cb2 == &s[0]);

    }
    assert(  b == cb1);
    assert(  b == cb2);
    assert(cb1 == cb2);
}


int main(int, char**)
{
    typedef std::string_view    string_view;
#ifndef TEST_HAS_NO_CHAR8_T
    typedef std::u8string_view  u8string_view;
#endif
    typedef std::u16string_view u16string_view;
    typedef std::u32string_view u32string_view;

    test(string_view   ());
    test(u16string_view());
    test(u32string_view());
    test(string_view   ( "123"));
#ifndef TEST_HAS_NO_CHAR8_T
    test(u8string_view{u8"123"});
#endif
#if TEST_STD_VER >= 11
    test(u16string_view{u"123"});
    test(u32string_view{U"123"});
#endif

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    typedef std::wstring_view   wstring_view;
    test(wstring_view  ());
    test(wstring_view  (L"123"));
#endif

#if TEST_STD_VER > 11
    {
    constexpr string_view       sv { "123", 3 };
#  ifndef TEST_HAS_NO_CHAR8_T
    constexpr u8string_view u8sv  {u8"123", 3 };
#endif
    constexpr u16string_view u16sv {u"123", 3 };
    constexpr u32string_view u32sv {U"123", 3 };

    static_assert (    *sv.begin() ==    sv[0], "" );
#  ifndef TEST_HAS_NO_CHAR8_T
    static_assert (  *u8sv.begin() ==  u8sv[0], "" );
#endif
    static_assert ( *u16sv.begin() == u16sv[0], "" );
    static_assert ( *u32sv.begin() == u32sv[0], "" );

    static_assert (    *sv.cbegin() ==    sv[0], "" );
#  ifndef TEST_HAS_NO_CHAR8_T
    static_assert (  *u8sv.cbegin() ==  u8sv[0], "" );
#endif
    static_assert ( *u16sv.cbegin() == u16sv[0], "" );
    static_assert ( *u32sv.cbegin() == u32sv[0], "" );

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        {
            constexpr wstring_view     wsv {L"123", 3 };
            static_assert (   *wsv.begin() ==   wsv[0], "" );
            static_assert (   *wsv.cbegin() ==   wsv[0], "" );
        }
#endif
    }
#endif // TEST_STD_VER > 11

  return 0;
}
