//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string_view>

// constexpr size_type find(charT c, size_type pos = 0) const;

#include <string_view>
#include <cassert>

#include "test_macros.h"
#include "constexpr_char_traits.h"

template <class S>
void
test(const S& s, typename S::value_type c, typename S::size_type pos,
     typename S::size_type x)
{
    assert(s.find(c, pos) == x);
    if (x != S::npos)
        assert(pos <= x && x + 1 <= s.size());
}

template <class S>
void
test(const S& s, typename S::value_type c, typename S::size_type x)
{
    assert(s.find(c) == x);
    if (x != S::npos)
        assert(0 <= x && x + 1 <= s.size());
}

int main(int, char**)
{
    {
    typedef std::string_view S;
    test(S(""), 'c', 0, S::npos);
    test(S(""), 'c', 1, S::npos);
    test(S("abcde"), 'c', 0, 2);
    test(S("abcde"), 'c', 1, 2);
    test(S("abcde"), 'c', 2, 2);
    test(S("abcde"), 'c', 4, S::npos);
    test(S("abcde"), 'c', 5, S::npos);
    test(S("abcde"), 'c', 6, S::npos);
    test(S("abcdeabcde"), 'c', 0, 2);
    test(S("abcdeabcde"), 'c', 1, 2);
    test(S("abcdeabcde"), 'c', 5, 7);
    test(S("abcdeabcde"), 'c', 9, S::npos);
    test(S("abcdeabcde"), 'c', 10, S::npos);
    test(S("abcdeabcde"), 'c', 11, S::npos);
    test(S("abcdeabcdeabcdeabcde"), 'c', 0, 2);
    test(S("abcdeabcdeabcdeabcde"), 'c', 1, 2);
    test(S("abcdeabcdeabcdeabcde"), 'c', 10, 12);
    test(S("abcdeabcdeabcdeabcde"), 'c', 19, S::npos);
    test(S("abcdeabcdeabcdeabcde"), 'c', 20, S::npos);
    test(S("abcdeabcdeabcdeabcde"), 'c', 21, S::npos);

    test(S(""), 'c', S::npos);
    test(S("abcde"), 'c', 2);
    test(S("abcdeabcde"), 'c', 2);
    test(S("abcdeabcdeabcdeabcde"), 'c', 2);
    }

#if TEST_STD_VER > 11
    {
    typedef std::basic_string_view<char, constexpr_char_traits<char>> SV;
    constexpr SV  sv1;
    constexpr SV  sv2 { "abcde", 5 };

    static_assert (sv1.find( 'c', 0 ) == SV::npos, "" );
    static_assert (sv1.find( 'c', 1 ) == SV::npos, "" );
    static_assert (sv2.find( 'c', 0 ) == 2, "" );
    static_assert (sv2.find( 'c', 1 ) == 2, "" );
    static_assert (sv2.find( 'c', 2 ) == 2, "" );
    static_assert (sv2.find( 'c', 3 ) == SV::npos, "" );
    static_assert (sv2.find( 'c', 4 ) == SV::npos, "" );
    }
#endif

  return 0;
}
