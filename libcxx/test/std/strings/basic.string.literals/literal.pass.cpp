// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

#include <string>
#include <cassert>

int main()
{
    using namespace std::literals::string_literals;

    static_assert ( std::is_same<decltype(   "Hi"s), std::string>::value, "" );
    static_assert ( std::is_same<decltype( u8"Hi"s), std::string>::value, "" );
    static_assert ( std::is_same<decltype(  L"Hi"s), std::wstring>::value, "" );
    static_assert ( std::is_same<decltype(  u"Hi"s), std::u16string>::value, "" );
    static_assert ( std::is_same<decltype(  U"Hi"s), std::u32string>::value, "" );

    std::string foo;
    std::wstring Lfoo;
    std::u16string ufoo;
    std::u32string Ufoo;

    foo  =   ""s;     assert( foo.size() == 0);
    foo  = u8""s;     assert( foo.size() == 0);
    Lfoo =  L""s;     assert(Lfoo.size() == 0);
    ufoo =  u""s;     assert(ufoo.size() == 0);
    Ufoo =  U""s;     assert(Ufoo.size() == 0);

    foo  =   " "s;     assert( foo.size() == 1);
    foo  = u8" "s;     assert( foo.size() == 1);
    Lfoo =  L" "s;     assert(Lfoo.size() == 1);
    ufoo =  u" "s;     assert(ufoo.size() == 1);
    Ufoo =  U" "s;     assert(Ufoo.size() == 1);

    foo  =   "ABC"s;     assert( foo ==   "ABC");   assert( foo == std::string   (  "ABC"));
    foo  = u8"ABC"s;     assert( foo == u8"ABC");   assert( foo == std::string   (u8"ABC"));
    Lfoo =  L"ABC"s;     assert(Lfoo ==  L"ABC");   assert(Lfoo == std::wstring  ( L"ABC"));
    ufoo =  u"ABC"s;     assert(ufoo ==  u"ABC");   assert(ufoo == std::u16string( u"ABC"));
    Ufoo =  U"ABC"s;     assert(Ufoo ==  U"ABC");   assert(Ufoo == std::u32string( U"ABC"));
}
