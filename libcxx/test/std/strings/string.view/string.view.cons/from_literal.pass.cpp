//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !stdlib=libc++ && (c++03 || c++11 || c++14)

// <string_view>

// constexpr basic_string_view(const _CharT* _s)
//    : __data (_s), __size(_Traits::length(_s)) {}

#include <string_view>
#include <string>
#include <cassert>

#include "test_macros.h"
#include "constexpr_char_traits.h"

template<typename CharT>
size_t StrLen ( const CharT *s ) {
    std::size_t retVal = 0;
    while ( *s != 0 ) { ++retVal; ++s; }
    return retVal;
    }

template<typename CharT>
void test ( const CharT *s ) {
    typedef std::basic_string_view<CharT> SV;
//  I'd love to do this, but it would require traits::length() to be noexcept
//  LIBCPP_ASSERT_NOEXCEPT(SV(s));

    SV sv1 ( s );
    assert ( sv1.size() == StrLen( s ));
    assert ( sv1.data() == s );
    }


int main(int, char**) {

    test ( "QBCDE" );
    test ( "A" );
    test ( "" );

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test ( L"QBCDE" );
    test ( L"A" );
    test ( L"" );
#endif

#if TEST_STD_VER >= 11
    test ( u"QBCDE" );
    test ( u"A" );
    test ( u"" );

    test ( U"QBCDE" );
    test ( U"A" );
    test ( U"" );
#endif

#if TEST_STD_VER > 11
    {
    constexpr std::basic_string_view<char, constexpr_char_traits<char>> sv1 ( "ABCDE" );
    static_assert ( sv1.size() == 5, "");
    }
#endif

  return 0;
}
