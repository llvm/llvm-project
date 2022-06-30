//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Aligned allocation is required by std::experimental::pmr, but it was not provided
// before macosx10.13 and as a result we get linker errors when deploying to older than
// macosx10.13.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// <experimental/string>

// namespace std { namespace experimental { namespace pmr {
// template <class Char, class Traits = ...>
// using basic_string =
//     ::std::basic_string<Char, Traits, polymorphic_allocator<Char>>
//
// typedef ... string
// typedef ... u16string
// typedef ... u32string
// typedef ... wstring
//
// }}} // namespace std::experimental::pmr

#include <experimental/string>
#include <experimental/memory_resource>
#include <type_traits>
#include <cassert>

#include "constexpr_char_traits.h"

#include "test_macros.h"

namespace pmr = std::experimental::pmr;

template <class Char, class PmrTypedef>
void test_string_typedef() {
    using StdStr = std::basic_string<Char, std::char_traits<Char>,
                                     pmr::polymorphic_allocator<Char>>;
    using PmrStr = pmr::basic_string<Char>;
    static_assert(std::is_same<StdStr, PmrStr>::value, "");
    static_assert(std::is_same<PmrStr, PmrTypedef>::value, "");
}

template <class Char, class Traits>
void test_basic_string_alias() {
    using StdStr = std::basic_string<Char, Traits,
                                     pmr::polymorphic_allocator<Char>>;
    using PmrStr = pmr::basic_string<Char, Traits>;
    static_assert(std::is_same<StdStr, PmrStr>::value, "");
}

int main(int, char**)
{
    {
        test_string_typedef<char,     pmr::string>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        test_string_typedef<wchar_t,  pmr::wstring>();
#endif
        test_string_typedef<char16_t, pmr::u16string>();
        test_string_typedef<char32_t, pmr::u32string>();
    }
    {
        test_basic_string_alias<char,    constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        test_basic_string_alias<wchar_t, constexpr_char_traits<wchar_t>>();
#endif
        test_basic_string_alias<char16_t, constexpr_char_traits<char16_t>>();
        test_basic_string_alias<char32_t, constexpr_char_traits<char32_t>>();
    }
    {
        // Check that std::basic_string has been included and is complete.
        pmr::string s;
        assert(s.get_allocator().resource() == pmr::get_default_resource());
    }

  return 0;
}
