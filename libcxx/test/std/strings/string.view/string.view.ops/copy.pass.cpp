//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// GCC's __builtin_strlen isn't constexpr yet
// XFAIL: (gcc-10 || gcc-11) && !(c++11 || c++14 || c++17)
// UNSUPPORTED: LIBCXX-DEBUG-FIXME

// <string_view>

// size_type copy(charT* s, size_type n, size_type pos = 0) const;

// Throws: out_of_range if pos > size().
// Remarks: Let rlen be the smaller of n and size() - pos.
// Requires: [s, s+rlen) is a valid range.
// Effects: Equivalent to std::copy_n(begin() + pos, rlen, s).
// Returns: rlen.


#include <string_view>
#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "test_macros.h"

template<typename CharT>
void test1 ( std::basic_string_view<CharT> sv, size_t n, size_t pos ) {
    const size_t rlen = std::min ( n, sv.size() - pos );

    CharT *dest1 = new CharT [rlen + 1];    dest1[rlen] = 0;
    CharT *dest2 = new CharT [rlen + 1];    dest2[rlen] = 0;

    if (pos > sv.size()) {
#ifndef TEST_HAS_NO_EXCEPTIONS
        try {
            sv.copy(dest1, n, pos);
            assert(false);
        } catch (const std::out_of_range&) {
        } catch (...) {
            assert(false);
        }
#endif
    } else {
        sv.copy(dest1, n, pos);
        std::copy_n(sv.begin() + pos, rlen, dest2);
        for ( size_t i = 0; i <= rlen; ++i )
            assert ( dest1[i] == dest2[i] );
    }
    delete [] dest1;
    delete [] dest2;
}


template<typename CharT>
void test ( const CharT *s ) {
    typedef std::basic_string_view<CharT> string_view_t;

    string_view_t sv1 ( s );

    test1(sv1,  0, 0);
    test1(sv1,  1, 0);
    test1(sv1, 20, 0);
    test1(sv1, sv1.size(), 0);
    test1(sv1, 20, string_view_t::npos);

    test1(sv1,   0, 3);
    test1(sv1,   2, 3);
    test1(sv1, 100, 3);
    test1(sv1, 100, string_view_t::npos);

    test1(sv1, sv1.size(), string_view_t::npos);

    test1(sv1, sv1.size() + 1, 0);
    test1(sv1, sv1.size() + 1, 1);
    test1(sv1, sv1.size() + 1, string_view_t::npos);
}

template<typename CharT>
TEST_CONSTEXPR_CXX20 bool test_constexpr_copy(const CharT *abcde, const CharT *ghijk, const CharT *bcdjk)
{
    CharT buf[6] = {};
    std::basic_string_view<CharT> lval(ghijk); lval.copy(buf, 6);
    std::basic_string_view<CharT>(abcde).copy(buf, 3, 1);
    assert(std::basic_string_view<CharT>(buf) == bcdjk);
    return true;
}

int main(int, char**) {
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

    test_constexpr_copy("ABCDE", "GHIJK", "BCDJK");
    test_constexpr_copy(L"ABCDE", L"GHIJK", L"BCDJK");
#if TEST_STD_VER >= 11
    test_constexpr_copy(u"ABCDE", u"GHIJK", u"BCDJK");
    test_constexpr_copy(U"ABCDE", U"GHIJK", U"BCDJK");
#endif
#if TEST_STD_VER >= 17
    test_constexpr_copy(u8"ABCDE", u8"GHIJK", u8"BCDJK");
#endif
#if TEST_STD_VER >= 20
    static_assert(test_constexpr_copy("ABCDE", "GHIJK", "BCDJK"));
    static_assert(test_constexpr_copy(L"ABCDE", L"GHIJK", L"BCDJK"));
    static_assert(test_constexpr_copy(u"ABCDE", u"GHIJK", u"BCDJK"));
    static_assert(test_constexpr_copy(U"ABCDE", U"GHIJK", U"BCDJK"));
    static_assert(test_constexpr_copy(u8"ABCDE", u8"GHIJK", u8"BCDJK"));
#endif

  return 0;
}
