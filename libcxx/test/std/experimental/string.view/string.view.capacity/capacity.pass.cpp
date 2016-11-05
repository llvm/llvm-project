//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// <string_view>

// [string.view.capacity], capacity
// constexpr size_type size()     const noexcept;
// constexpr size_type length()   const noexcept;
// constexpr size_type max_size() const noexcept;
// constexpr bool empty()         const noexcept;

#include <experimental/string_view>
#include <cassert>

#include "test_macros.h"

template<typename SV>
void test1 () {
#if TEST_STD_VER > 11
    {
    constexpr SV sv1;
    static_assert ( sv1.size() == 0, "" );
    static_assert ( sv1.empty(), "");
    static_assert ( sv1.size() == sv1.length(), "" );
    static_assert ( sv1.max_size() > sv1.size(), "");
    }
#endif

    {
    SV sv1;
    assert ( sv1.size() == 0 );
    assert ( sv1.empty());
    assert ( sv1.size() == sv1.length());
    assert ( sv1.max_size() > sv1.size());
    }
}

template<typename CharT>
void test2 ( const CharT *s, size_t len ) {
    {
    std::experimental::basic_string_view<CharT> sv1 ( s );
    assert ( sv1.size() == len );
    assert ( sv1.data() == s );
    assert ( sv1.empty() == (len == 0));
    assert ( sv1.size() == sv1.length());
    assert ( sv1.max_size() > sv1.size());
    }
}

int main () {
    typedef std::experimental::string_view    string_view;
    typedef std::experimental::u16string_view u16string_view;
    typedef std::experimental::u32string_view u32string_view;
    typedef std::experimental::wstring_view   wstring_view;

    test1<string_view> ();
    test1<u16string_view> ();
    test1<u32string_view> ();
    test1<wstring_view> ();

    test2 ( "ABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDE", 105 );
    test2 ( "ABCDE", 5 );
    test2 ( "a", 1 );
    test2 ( "", 0 );

    test2 ( L"ABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDE", 105 );
    test2 ( L"ABCDE", 5 );
    test2 ( L"a", 1 );
    test2 ( L"", 0 );

#if TEST_STD_VER >= 11
    test2 ( u"ABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDE", 105 );
    test2 ( u"ABCDE", 5 );
    test2 ( u"a", 1 );
    test2 ( u"", 0 );

    test2 ( U"ABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDEABCDE", 105 );
    test2 ( U"ABCDE", 5 );
    test2 ( U"a", 1 );
    test2 ( U"", 0 );
#endif
}
