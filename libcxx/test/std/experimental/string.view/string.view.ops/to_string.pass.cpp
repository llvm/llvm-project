//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// <string_view>

//  template<class Allocator>
//  explicit operator basic_string<_CharT, _Traits, Allocator> () const;
//  template<class _CharT, class _Traits = char_traits<_CharT>, class Allocator = allocator<_CharT> >
//  basic_string<_CharT, _Traits, Allocator> to_string (
//          basic_string_view<_CharT, _Traits> _sv, const Allocator& _a = Allocator()) const;

#include <experimental/string_view>
#include <cassert>
#include "min_allocator.h"

template<typename CharT>
void test ( const CharT *s ) {
    typedef std::basic_string<CharT> String ;
    {
    const std::experimental::basic_string_view<CharT> sv1 ( s );
    String                                            str1 = (String) sv1;

    assert ( sv1.size() == str1.size ());
    assert ( std::char_traits<CharT>::compare ( sv1.data(), str1.data(),  sv1.size()) == 0 );

#if __cplusplus >= 201103L
    auto str2 = sv1.to_string(min_allocator<CharT>());
    assert ( sv1.size() == str2.size ());
    assert ( std::char_traits<CharT>::compare ( sv1.data(), str2.data(), sv1.size()) == 0 );
#endif
    }

    {
    const std::experimental::basic_string_view<CharT> sv1;
    String                                            str1 = (String) sv1;

    assert ( sv1.size() == 0);
    assert ( sv1.size() == str1.size ());

#if __cplusplus >= 201103L
    auto str2 = sv1.to_string(min_allocator<CharT>());
    assert ( sv1.size() == str2.size ());
#endif
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

#if __cplusplus >= 201103L
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
