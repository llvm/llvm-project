//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iosfwd>

#include <iosfwd>

#include "test_macros.h"

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
#   include <cwchar>
#endif

template <class Ptr> void test()
{
    Ptr p = 0;
    ((void)p); // Prevent unused warning
}

int main(int, char**)
{
    test<std::char_traits<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::char_traits<wchar_t>*       >();
#endif

    test<std::basic_ios<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_ios<wchar_t>*       >();
#endif
    test<std::basic_ios<unsigned short>*>();

    test<std::basic_streambuf<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_streambuf<wchar_t>*       >();
#endif
    test<std::basic_streambuf<unsigned short>*>();

    test<std::basic_istream<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_istream<wchar_t>*       >();
#endif
    test<std::basic_istream<unsigned short>*>();

    test<std::basic_ostream<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_ostream<wchar_t>*       >();
#endif
    test<std::basic_ostream<unsigned short>*>();

    test<std::basic_iostream<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_iostream<wchar_t>*       >();
#endif
    test<std::basic_iostream<unsigned short>*>();

    test<std::basic_stringbuf<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_stringbuf<wchar_t>*       >();
#endif
    test<std::basic_stringbuf<unsigned short>*>();

    test<std::basic_istringstream<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_istringstream<wchar_t>*       >();
#endif
    test<std::basic_istringstream<unsigned short>*>();

    test<std::basic_ostringstream<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_ostringstream<wchar_t>*       >();
#endif
    test<std::basic_ostringstream<unsigned short>*>();

    test<std::basic_stringstream<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_stringstream<wchar_t>*       >();
#endif
    test<std::basic_stringstream<unsigned short>*>();

#if TEST_STD_VER >= 23
    test<std::basic_spanbuf<unsigned short>*>();
    test<std::basic_spanbuf<char>*>();
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_spanbuf<wchar_t>*>();
#  endif
    test<std::basic_ispanstream<unsigned short>*>();
    test<std::basic_ispanstream<char>*>();
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_ispanstream<wchar_t>*>();
#  endif
    test<std::basic_ospanstream<unsigned short>*>();
    test<std::basic_ospanstream<char>*>();
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_ospanstream<wchar_t>*>();
#  endif
    test<std::basic_spanstream<char>*>();
    test<std::basic_spanstream<unsigned short>*>();
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_spanstream<wchar_t>*>();
#  endif
#endif // TEST_STD_VER >= 23

    test<std::basic_filebuf<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_filebuf<wchar_t>*       >();
#endif
    test<std::basic_filebuf<unsigned short>*>();

    test<std::basic_ifstream<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_ifstream<wchar_t>*       >();
#endif
    test<std::basic_ifstream<unsigned short>*>();

    test<std::basic_ofstream<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_ofstream<wchar_t>*       >();
#endif
    test<std::basic_ofstream<unsigned short>*>();

    test<std::basic_fstream<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::basic_fstream<wchar_t>*       >();
#endif
    test<std::basic_fstream<unsigned short>*>();

    test<std::istreambuf_iterator<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::istreambuf_iterator<wchar_t>*       >();
#endif
    test<std::istreambuf_iterator<unsigned short>*>();

    test<std::ostreambuf_iterator<char>*          >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::ostreambuf_iterator<wchar_t>*       >();
#endif
    test<std::ostreambuf_iterator<unsigned short>*>();

    test<std::ios* >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::wios*>();
#endif

    test<std::streambuf*>();
    test<std::istream*  >();
    test<std::ostream*  >();
    test<std::iostream* >();

    test<std::stringbuf*    >();
    test<std::istringstream*>();
    test<std::ostringstream*>();
    test<std::stringstream* >();

#if TEST_STD_VER >= 23
    test<std::spanbuf*>();
    test<std::ispanstream*>();
    test<std::ospanstream*>();
    test<std::spanstream*>();
#endif // TEST_STD_VER >= 23

    test<std::filebuf* >();
    test<std::ifstream*>();
    test<std::ofstream*>();
    test<std::fstream* >();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::wstreambuf*>();
    test<std::wistream*  >();
    test<std::wostream*  >();
    test<std::wiostream* >();

    test<std::wstringbuf*    >();
    test<std::wistringstream*>();
    test<std::wostringstream*>();
    test<std::wstringstream* >();

#  if TEST_STD_VER >= 23
    test<std::wspanbuf*>();
    test<std::wispanstream*>();
    test<std::wospanstream*>();
    test<std::wspanstream*>();
#  endif // TEST_STD_VER >= 23

    test<std::wfilebuf* >();
    test<std::wifstream*>();
    test<std::wofstream*>();
    test<std::wfstream* >();
#endif

    test<std::fpos<std::mbstate_t>*>();
    test<std::streampos*           >();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::wstreampos*          >();
#endif

  return 0;
}
