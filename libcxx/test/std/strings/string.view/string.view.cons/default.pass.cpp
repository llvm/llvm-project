//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string_view>

// constexpr basic_string_view () noexcept;

#include <string_view>
#include <cassert>

#include "test_macros.h"

template<typename T>
void test () {
#if TEST_STD_VER > 11
    {
    ASSERT_NOEXCEPT(T());

    constexpr T sv1;
    static_assert ( sv1.size() == 0, "" );
    static_assert ( sv1.empty(), "");
    }
#endif

    {
    T sv1;
    assert ( sv1.size() == 0 );
    assert ( sv1.empty());
    }
}

int main(int, char**) {
    test<std::string_view> ();
    test<std::u16string_view> ();
#if defined(__cpp_lib_char8_t) && __cpp_lib_char8_t >= 201811L
    test<std::u8string_view> ();
#endif
    test<std::u32string_view> ();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    test<std::wstring_view> ();
#endif

  return 0;
}
