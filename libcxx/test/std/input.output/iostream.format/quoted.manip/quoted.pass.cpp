//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <iomanip>

// quoted

#include <iomanip>
#include <sstream>
#include <string>
#include <cassert>

#include "test_macros.h"

template <class CharT, class Traits>
bool is_skipws ( const std::basic_istream<CharT, Traits>& is ) {
    return ( is.flags() & std::ios_base::skipws ) != 0;
}

template <class CharT, class Traits = std::char_traits<CharT>>
void both_ways ( const CharT *p ) {
    std::basic_string<CharT, Traits> str(p);
    auto q = std::quoted(str);

    std::basic_stringstream<CharT, Traits> ss;
    bool skippingws = is_skipws ( ss );
    ((void)skippingws); // Prevent unused warning
    ss << q;
    ss >> q;
}

template <class CharT, class Traits = std::char_traits<CharT>>
void round_trip ( const CharT *p ) {
    std::basic_stringstream<CharT, Traits> ss;
    bool skippingws = is_skipws ( ss );

    ss << std::quoted(p);
    std::basic_string<CharT, Traits> s;
    ss >> std::quoted(s);
    assert ( s == p );
    assert ( skippingws == is_skipws ( ss ));
}


template <class CharT, class Traits = std::char_traits<CharT>>
void round_trip_ws ( const CharT *p ) {
    std::basic_stringstream<CharT, Traits> ss;
    std::noskipws ( ss );
    bool skippingws = is_skipws ( ss );

    ss << std::quoted(p);
    std::basic_string<CharT, Traits> s;
    ss >> std::quoted(s);
    assert ( s == p );
    assert ( skippingws == is_skipws ( ss ));
}

template <class CharT, class Traits = std::char_traits<CharT>>
void round_trip_d ( const CharT *p, char delim ) {
    std::basic_stringstream<CharT, Traits> ss;
    CharT d(delim);

    ss << std::quoted(p, d);
    std::basic_string<CharT, Traits> s;
    ss >> std::quoted(s, d);
    assert ( s == p );
}

template <class CharT, class Traits = std::char_traits<CharT>>
void round_trip_e ( const CharT *p, char escape ) {
    std::basic_stringstream<CharT, Traits> ss;
    CharT e(escape);

    ss << std::quoted(p, CharT('"'), e );
    std::basic_string<CharT, Traits> s;
    ss >> std::quoted(s, CharT('"'), e );
    assert ( s == p );
}


template <class CharT, class Traits = std::char_traits<CharT>>
std::basic_string<CharT, Traits> quote ( const CharT *p, char delim='"', char escape='\\' ) {
    std::basic_stringstream<CharT, Traits> ss;
    CharT d(delim);
    CharT e(escape);
    ss << std::quoted(p, d, e);
    std::basic_string<CharT, Traits> s;
    ss >> s;    // no quote
    return s;
}

template <class CharT, class Traits = std::char_traits<CharT>>
std::basic_string<CharT, Traits> unquote ( const CharT *p, char delim='"', char escape='\\' ) {
    std::basic_stringstream<CharT, Traits> ss;
    ss << p;

    CharT d(delim);
    CharT e(escape);
    std::basic_string<CharT, Traits> s;
    ss >> std::quoted(s, d, e);
    return s;
}

void test_padding () {
    {
    std::stringstream ss;
    ss << std::left << std::setw(10) << std::setfill('!') << std::quoted("abc", '`');
    assert ( ss.str() == "`abc`!!!!!" );
    }

    {
    std::stringstream ss;
    ss << std::right << std::setw(10) << std::setfill('!') << std::quoted("abc", '`');
    assert ( ss.str() == "!!!!!`abc`" );
    }
}


int main(int, char**)
{
    both_ways ( "" );   // This is a compilation check

    round_trip    (  "" );
    round_trip_ws (  "" );
    round_trip_d  (  "", 'q' );
    round_trip_e  (  "", 'q' );

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    round_trip    ( L"" );
    round_trip_ws ( L"" );
    round_trip_d  ( L"", 'q' );
    round_trip_e  ( L"", 'q' );
#endif

    round_trip    (  "Hi" );
    round_trip_ws (  "Hi" );
    round_trip_d  (  "Hi", '!' );
    round_trip_e  (  "Hi", '!' );
    assert ( quote ( "Hi", '!' ) == "!Hi!" );
    assert ( quote ( "Hi!", '!' ) == R"(!Hi\!!)" );

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    round_trip    ( L"Hi" );
    round_trip_ws ( L"Hi" );
    round_trip_d  ( L"Hi", '!' );
    round_trip_e  ( L"Hi", '!' );
    assert ( quote ( L"Hi", '!' )  == L"!Hi!" );
    assert ( quote ( L"Hi!", '!' ) == LR"(!Hi\!!)" );
#endif

    round_trip    (  "Hi Mom" );
    round_trip_ws (  "Hi Mom" );
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    round_trip    ( L"Hi Mom" );
    round_trip_ws ( L"Hi Mom" );
#endif

    assert ( quote (  "" )  ==  "\"\"" );
    assert ( quote (  "a" ) ==  "\"a\"" );
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    assert ( quote ( L"" )  == L"\"\"" );
    assert ( quote ( L"a" ) == L"\"a\"" );
#endif

    // missing end quote - must not hang
    assert ( unquote (  "\"abc" ) ==  "abc" );
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    assert ( unquote ( L"\"abc" ) == L"abc" );
#endif

    assert ( unquote (  "abc" ) == "abc" ); // no delimiter
    assert ( unquote (  "abc def" ) ==  "abc" ); // no delimiter
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    assert ( unquote ( L"abc" ) == L"abc" ); // no delimiter
    assert ( unquote ( L"abc def" ) == L"abc" ); // no delimiter
#endif

    assert ( unquote (  "" ) ==  "" ); // nothing there
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    assert ( unquote ( L"" ) == L"" ); // nothing there
#endif

    test_padding ();

    return 0;
}
