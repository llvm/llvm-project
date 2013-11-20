//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iomanip>

// quoted

#include <iomanip>
#include <sstream>
#include <string>
#include <cassert>

#if _LIBCPP_STD_VER > 11

bool is_skipws ( const std::istream *is ) {
    return ( is->flags() & std::ios_base::skipws ) != 0;
    }


bool is_skipws ( const std::wistream *is ) {
    return ( is->flags() & std::ios_base::skipws ) != 0;
    }

void both_ways ( const char *p ) {
	std::string str(p);
	auto q = std::quoted(str);

    std::stringstream ss;
    bool skippingws = is_skipws ( &ss );
	ss << q;
	ss >> q;
    }

void round_trip ( const char *p ) {
    std::stringstream ss;
    bool skippingws = is_skipws ( &ss );
    ss << std::quoted(p);
    std::string s;
    ss >> std::quoted(s);
    assert ( s == p );
    assert ( skippingws == is_skipws ( &ss ));
    }

void round_trip_ws ( const char *p ) {
    std::stringstream ss;
    std::noskipws ( ss );
    bool skippingws = is_skipws ( &ss );
    ss << std::quoted(p);
    std::string s;
    ss >> std::quoted(s);
    assert ( s == p );
    assert ( skippingws == is_skipws ( &ss ));
    }

void round_trip_d ( const char *p, char delim ) {
    std::stringstream ss;
    ss << std::quoted(p, delim);
    std::string s;
    ss >> std::quoted(s, delim);
    assert ( s == p );
    }

void round_trip_e ( const char *p, char escape ) {
    std::stringstream ss;
    ss << std::quoted(p, '"', escape );
    std::string s;
    ss >> std::quoted(s, '"', escape );
    assert ( s == p );
    }



std::string quote ( const char *p, char delim='"', char escape='\\' ) {
    std::stringstream ss;
    ss << std::quoted(p, delim, escape);
    std::string s;
    ss >> s;    // no quote
    return s;
}

std::string unquote ( const char *p, char delim='"', char escape='\\' ) {
    std::stringstream ss;
    ss << p;
    std::string s;
    ss >> std::quoted(s, delim, escape);
    return s;
}


void round_trip ( const wchar_t *p ) {
    std::wstringstream ss;
    bool skippingws = is_skipws ( &ss );
    ss << std::quoted(p);
    std::wstring s;
    ss >> std::quoted(s);
    assert ( s == p );
    assert ( skippingws == is_skipws ( &ss ));
    }
    

void round_trip_ws ( const wchar_t *p ) {
    std::wstringstream ss;
    std::noskipws ( ss );
    bool skippingws = is_skipws ( &ss );
    ss << std::quoted(p);
    std::wstring s;
    ss >> std::quoted(s);
    assert ( s == p );
    assert ( skippingws == is_skipws ( &ss ));
    }

void round_trip_d ( const wchar_t *p, wchar_t delim ) {
    std::wstringstream ss;
    ss << std::quoted(p, delim);
    std::wstring s;
    ss >> std::quoted(s, delim);
    assert ( s == p );
    }

void round_trip_e ( const wchar_t *p, wchar_t escape ) {
    std::wstringstream ss;
    ss << std::quoted(p, wchar_t('"'), escape );
    std::wstring s;
    ss >> std::quoted(s, wchar_t('"'), escape );
    assert ( s == p );
    }


std::wstring quote ( const wchar_t *p, wchar_t delim='"', wchar_t escape='\\' ) {
    std::wstringstream ss;
    ss << std::quoted(p, delim, escape);
    std::wstring s;
    ss >> s;    // no quote
    return s;
}

std::wstring unquote ( const wchar_t *p, wchar_t delim='"', wchar_t escape='\\' ) {
    std::wstringstream ss;
    ss << p;
    std::wstring s;
    ss >> std::quoted(s, delim, escape);
    return s;
}

int main()
{
    both_ways ( "" );   // This is a compilation check

    round_trip    (  "" );
    round_trip_ws (  "" );
    round_trip_d  (  "", 'q' );
    round_trip_e  (  "", 'q' );
    
    round_trip    ( L"" );
    round_trip_ws ( L"" );
    round_trip_d  ( L"", 'q' );
    round_trip_e  ( L"", 'q' );
    
    round_trip    (  "Hi" );
    round_trip_ws (  "Hi" );
    round_trip_d  (  "Hi", '!' );
    round_trip_e  (  "Hi", '!' );
    assert ( quote ( "Hi", '!' ) == "!Hi!" );
    assert ( quote ( "Hi!", '!' ) == R"(!Hi\!!)" );
    
    round_trip    ( L"Hi" );
    round_trip_ws ( L"Hi" );
    round_trip_d  ( L"Hi", '!' );
    round_trip_e  ( L"Hi", '!' );
    assert ( quote ( L"Hi", '!' )  == L"!Hi!" );
    assert ( quote ( L"Hi!", '!' ) == LR"(!Hi\!!)" );
    
    round_trip    (  "Hi Mom" );
    round_trip_ws (  "Hi Mom" );
    round_trip    ( L"Hi Mom" );
    round_trip_ws ( L"Hi Mom" );
    
    assert ( quote (  "" )  ==  "\"\"" );
    assert ( quote ( L"" )  == L"\"\"" );
    assert ( quote (  "a" ) ==  "\"a\"" );
    assert ( quote ( L"a" ) == L"\"a\"" );

//  missing end quote - must not hang
    assert ( unquote (  "\"abc" ) ==  "abc" );
    assert ( unquote ( L"\"abc" ) == L"abc" );
    
    assert ( unquote (  "abc" ) == "abc" ); // no delimiter
    assert ( unquote ( L"abc" ) == L"abc" ); // no delimiter
    assert ( unquote (  "abc def" ) ==  "abc" ); // no delimiter
    assert ( unquote ( L"abc def" ) == L"abc" ); // no delimiter

    assert ( unquote (  "" ) ==  "" ); // nothing there
    assert ( unquote ( L"" ) == L"" ); // nothing there
    }

#else
int main() {}
#endif
