//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// template<class charT, class traits>
//   constexpr bool operator>=(basic_string_view<charT,traits> lhs,
//                  basic_string_view<charT,traits> rhs);

#include <string_view>
#include <cassert>

#include "test_macros.h"
#include "constexpr_char_traits.h"

template <class S>
void
test(const S& lhs, const S& rhs, bool x, bool y)
{
    assert((lhs >= rhs) == x);
    assert((rhs >= lhs) == y);
}

int main(int, char**)
{
    {
    typedef std::string_view S;
    test(S(""), S(""), true, true);
    test(S(""), S("abcde"), false, true);
    test(S(""), S("abcdefghij"), false, true);
    test(S(""), S("abcdefghijklmnopqrst"), false, true);
    test(S("abcde"), S(""), true, false);
    test(S("abcde"), S("abcde"), true, true);
    test(S("abcde"), S("abcdefghij"), false, true);
    test(S("abcde"), S("abcdefghijklmnopqrst"), false, true);
    test(S("abcdefghij"), S(""), true, false);
    test(S("abcdefghij"), S("abcde"), true, false);
    test(S("abcdefghij"), S("abcdefghij"), true, true);
    test(S("abcdefghij"), S("abcdefghijklmnopqrst"), false, true);
    test(S("abcdefghijklmnopqrst"), S(""), true, false);
    test(S("abcdefghijklmnopqrst"), S("abcde"), true, false);
    test(S("abcdefghijklmnopqrst"), S("abcdefghij"), true, false);
    test(S("abcdefghijklmnopqrst"), S("abcdefghijklmnopqrst"), true, true);
    }

#if TEST_STD_VER > 11
    {
    typedef std::basic_string_view<char, constexpr_char_traits<char>> SV;
    constexpr SV  sv1;
    constexpr SV  sv2 { "abcde", 5 };

    static_assert (  sv1 >= sv1,  "" );
    static_assert (  sv2 >= sv2,  "" );

    static_assert (!(sv1 >= sv2), "" );
    static_assert (  sv2 >= sv1,  "" );
    }
#endif

  return 0;
}
