//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string_view>

// constexpr const _CharT& back();

#include <string_view>
#include <cassert>

#include "test_macros.h"

template <typename CharT>
bool test ( const CharT *s, size_t len ) {
    typedef std::basic_string_view<CharT> SV;
    SV sv ( s, len );
    ASSERT_SAME_TYPE(decltype(sv.front()), typename SV::const_reference);
    LIBCPP_ASSERT_NOEXCEPT(   sv.front());
    assert ( sv.length() == len );
    assert ( sv.front() == s[0] );
    return &sv.front() == s;
    }

int main(int, char**) {
    assert ( test ( "ABCDE", 5 ));
    assert ( test ( "a", 1 ));

    assert ( test ( L"ABCDE", 5 ));
    assert ( test ( L"a", 1 ));

#if TEST_STD_VER >= 11
    assert ( test ( u"ABCDE", 5 ));
    assert ( test ( u"a", 1 ));

    assert ( test ( U"ABCDE", 5 ));
    assert ( test ( U"a", 1 ));
#endif

#if TEST_STD_VER >= 11
    {
    constexpr std::basic_string_view<char> sv ( "ABC", 2 );
    static_assert ( sv.length() ==  2,  "" );
    static_assert ( sv.front()  == 'A', "" );
    }
#endif

  return 0;
}
