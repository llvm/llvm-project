//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !stdlib=libc++ && (c++03 || c++11 || c++14)

// <string_view>

// const_iterator rbegin() const;

#include <string_view>
#include <cassert>

#include "test_macros.h"

template <class S>
void
test(S s)
{
    const S& cs = s;
    typename S::reverse_iterator b = s.rbegin();
    typename S::const_reverse_iterator cb1 = cs.rbegin();
    typename S::const_reverse_iterator cb2 = s.crbegin();
    if (!s.empty())
    {
        const size_t last = s.size() - 1;
        assert(   *b ==  s[last]);
        assert(  &*b == &s[last]);
        assert( *cb1 ==  s[last]);
        assert(&*cb1 == &s[last]);
        assert( *cb2 ==  s[last]);
        assert(&*cb2 == &s[last]);

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

#if TEST_STD_VER > 14
    {
    constexpr string_view       sv { "123", 3 };
#  ifndef TEST_HAS_NO_CHAR8_T
    constexpr u8string_view u8sv  {u8"123", 3 };
#endif
    constexpr u16string_view u16sv {u"123", 3 };
    constexpr u32string_view u32sv {U"123", 3 };

    static_assert (    *sv.rbegin() ==    sv[2], "" );
#  ifndef TEST_HAS_NO_CHAR8_T
    static_assert (  *u8sv.rbegin() ==  u8sv[2], "" );
#endif
    static_assert ( *u16sv.rbegin() == u16sv[2], "" );
    static_assert ( *u32sv.rbegin() == u32sv[2], "" );

    static_assert (    *sv.crbegin() ==    sv[2], "" );
#  ifndef TEST_HAS_NO_CHAR8_T
    static_assert (  *u8sv.crbegin() == u8sv[2], "" );
#endif
    static_assert ( *u16sv.crbegin() == u16sv[2], "" );
    static_assert ( *u32sv.crbegin() == u32sv[2], "" );

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        {
            constexpr wstring_view     wsv {L"123", 3 };
            static_assert (   *wsv.rbegin() ==   wsv[2], "" );
            static_assert (   *wsv.crbegin() ==   wsv[2], "" );
        }
#endif
    }
#endif // TEST_STD_VER > 14

  return 0;
}
