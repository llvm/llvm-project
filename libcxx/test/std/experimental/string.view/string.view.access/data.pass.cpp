//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// <string_view>

// constexpr const _CharT* data() const noexcept;

#include <experimental/string_view>
#include <cassert>

template <typename CharT>
void test ( const CharT *s, size_t len ) {
    std::experimental::basic_string_view<CharT> sv ( s, len );
    assert ( sv.length() == len );
    assert ( sv.data() == s );
    }

int main () {
    test ( "ABCDE", 5 );
    test ( "a", 1 );

    test ( L"ABCDE", 5 );
    test ( L"a", 1 );

#if __cplusplus >= 201103L
    test ( u"ABCDE", 5 );
    test ( u"a", 1 );

    test ( U"ABCDE", 5 );
    test ( U"a", 1 );
#endif

#if _LIBCPP_STD_VER > 11
    {
    constexpr const char *s = "ABC";
    constexpr std::experimental::basic_string_view<char> sv( s, 2 );
    static_assert( sv.length() ==  2,  "" );
    static_assert( sv.data() == s, "" );
    }
#endif
}
