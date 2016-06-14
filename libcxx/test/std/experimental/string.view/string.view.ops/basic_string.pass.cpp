//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// <string_view>

// template<class _Allocator>
// explicit operator basic_string<_CharT, _Traits, _Allocator>() const

#include <experimental/string_view>
#include <cassert>

#include "test_macros.h"

template<typename CharT>
void test ( const CharT *s ) {
    typedef std::experimental::basic_string_view<CharT> string_view_t;
    typedef std::basic_string<CharT> string_t;

    {
    string_view_t sv1 ( s );
    string_t      str = (string_t) sv1;

    assert ( sv1.size() == str.size ());
    assert ( std::char_traits<CharT>::compare ( sv1.data(), str.data(), sv1.size()) == 0 );
    }

    {
    string_view_t sv1;
    string_t      str = (string_t) sv1;

    assert ( sv1.size() ==  0);
    assert ( sv1.size() == str.size ());
    }
}

int main () {
    test ( "ABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDE" );
    test ( "ABCDE");
    test ( "a" );
    test ( "" );

    test ( L"ABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDE" );
    test ( L"ABCDE" );
    test ( L"a" );
    test ( L"" );

#if TEST_STD_VER >= 11
    test ( u"ABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDE" );
    test ( u"ABCDE" );
    test ( u"a" );
    test ( u"" );

    test ( U"ABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDE" );
    test ( U"ABCDE" );
    test ( U"a" );
    test ( U"" );
#endif
}
