//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !stdlib=libc++ && (c++03 || c++11 || c++14)

// <string_view>

// constexpr basic_string_view& operator=(const basic_string_view &) noexcept = default;

#include <string_view>
#include <cassert>

#include "test_macros.h"

template<typename T>
#if TEST_STD_VER > 11
constexpr
#endif
bool test (T sv0)
    {
    T sv1;
    sv1 = sv0;
//  We can't just say "sv0 == sv1" here because string_view::compare
//  isn't constexpr until C++17, and we want to support back to C++14
    return sv0.size() == sv1.size() && sv0.data() == sv1.data();
    }

int main(int, char**) {

    assert( test<std::string_view>    (  "1234"));
#ifndef TEST_HAS_NO_CHAR8_T
    assert( test<std::u8string_view>  (u8"1234"));
#endif
#if TEST_STD_VER >= 11
    assert( test<std::u16string_view> ( u"1234"));
    assert( test<std::u32string_view> ( U"1234"));
#endif
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    assert( test<std::wstring_view>   ( L"1234"));
#endif

#if TEST_STD_VER > 11
    static_assert( test<std::string_view>    ({  "abc", 3}), "");
#  ifndef TEST_HAS_NO_CHAR8_T
    static_assert( test<std::u8string_view>  ({u8"abc", 3}), "");
#   endif
    static_assert( test<std::u16string_view> ({ u"abc", 3}), "");
    static_assert( test<std::u32string_view> ({ U"abc", 3}), "");
#   ifndef TEST_HAS_NO_WIDE_CHARACTERS
    static_assert( test<std::wstring_view>   ({ L"abc", 3}), "");
#   endif
#endif

  return 0;
}
