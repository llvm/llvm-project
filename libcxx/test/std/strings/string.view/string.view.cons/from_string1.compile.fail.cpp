//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !stdlib=libc++ && (c++03 || c++11 || c++14)

// <string_view>

// template<class Allocator>
// basic_string_view(const basic_string<_CharT, _Traits, Allocator>& _str) noexcept

#include <string_view>
#include <string>
#include <cassert>

struct dummy_char_traits : public std::char_traits<char> {};

int main(int, char**) {
    using string_view = std::basic_string_view<char>;
    using string      = std::              basic_string     <char, dummy_char_traits>;

    {
    string s{"QBCDE"};
    string_view sv1 ( s );
    assert ( sv1.size() == s.size());
    assert ( sv1.data() == s.data());
    }

  return 0;
}
