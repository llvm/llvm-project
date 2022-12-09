//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !stdlib=libc++ && (c++03 || c++11 || c++14)

// <string_view>

// constexpr const _CharT& operator[](size_type _pos) const;

#include <string_view>
#include <cassert>

#include "test_macros.h"

template <typename CharT>
void test ( const CharT *s, size_t len ) {
    typedef std::basic_string_view<CharT> SV;
    SV sv ( s, len );
    ASSERT_SAME_TYPE(decltype(sv[0]), typename SV::const_reference);
    LIBCPP_ASSERT_NOEXCEPT(   sv[0]);
    assert ( sv.length() == len );
    for ( size_t i = 0; i < len; ++i ) {
        assert ( sv[i] == s[i] );
        assert ( &sv[i] == s + i );
        }
    }

int main(int, char**) {
    test ( "ABCDE", 5 );
    test ( "a", 1 );

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test ( L"ABCDE", 5 );
    test ( L"a", 1 );
#endif

#if TEST_STD_VER >= 11
    test ( u"ABCDE", 5 );
    test ( u"a", 1 );

    test ( U"ABCDE", 5 );
    test ( U"a", 1 );
#endif

#if TEST_STD_VER > 11
    {
    constexpr std::basic_string_view<char> sv ( "ABC", 2 );
    static_assert ( sv.length() ==  2,  "" );
    static_assert ( sv[0]  == 'A', "" );
    static_assert ( sv[1]  == 'B', "" );
    }
#endif

  return 0;
}
