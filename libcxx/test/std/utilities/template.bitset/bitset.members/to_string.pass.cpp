//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test:

// template <class charT, class traits, class Allocator>
// basic_string<charT, traits, Allocator>
// to_string(charT zero = charT('0'), charT one = charT('1')) const;
//
// template <class charT, class traits>
// basic_string<charT, traits, allocator<charT> > to_string() const;
//
// template <class charT>
// basic_string<charT, char_traits<charT>, allocator<charT> > to_string() const;
//
// basic_string<char, char_traits<char>, allocator<char> > to_string() const;

#include <bitset>
#include <cassert>
#include <cstddef>
#include <memory> // for std::allocator
#include <string>
#include <vector>

#include "../bitset_test_cases.h"
#include "test_macros.h"

template <class CharT, std::size_t N>
void check_equal(std::basic_string<CharT> const& s, std::bitset<N> const& b, CharT zero, CharT one) {
    assert(s.size() == b.size());
    for (std::size_t i = 0; i < b.size(); ++i) {
        if (b[i]) {
            assert(s[b.size() - 1 - i] == one);
        } else {
            assert(s[b.size() - 1 - i] == zero);
        }
    }
}

template <std::size_t N>
void test_to_string() {
    std::vector<std::bitset<N> > const cases = get_test_cases<N>();
    for (std::size_t c = 0; c != cases.size(); ++c) {
        std::bitset<N> const v = cases[c];
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        {
            std::wstring s = v.template to_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> >();
            check_equal(s, v, L'0', L'1');
        }
        {
            std::wstring s = v.template to_string<wchar_t, std::char_traits<wchar_t> >();
            check_equal(s, v, L'0', L'1');
        }
#endif
        {
            std::string s = v.template to_string<char>();
            check_equal(s, v, '0', '1');
        }
        {
            std::string s = v.to_string();
            check_equal(s, v, '0', '1');
        }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        {
            std::wstring s = v.template to_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> >('0');
            check_equal(s, v, L'0', L'1');
        }
        {
            std::wstring s = v.template to_string<wchar_t, std::char_traits<wchar_t> >('0');
            check_equal(s, v, L'0', L'1');
        }
#endif
        {
            std::string s = v.template to_string<char>('0');
            check_equal(s, v, '0', '1');
        }
        {
            std::string s = v.to_string('0');
            check_equal(s, v, '0', '1');
        }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
        {
            std::wstring s = v.template to_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> >('0', '1');
            check_equal(s, v, L'0', L'1');
        }
        {
            std::wstring s = v.template to_string<wchar_t, std::char_traits<wchar_t> >('0', '1');
            check_equal(s, v, L'0', L'1');
        }
#endif
        {
            std::string s = v.template to_string<char>('0', '1');
            check_equal(s, v, '0', '1');
        }
        {
            std::string s = v.to_string('0', '1');
            check_equal(s, v, '0', '1');
        }
        {
            std::string s = v.to_string('x', 'y');
            check_equal(s, v, 'x', 'y');
        }
    }
}

int main(int, char**) {
    test_to_string<0>();
    test_to_string<1>();
    test_to_string<31>();
    test_to_string<32>();
    test_to_string<33>();
    test_to_string<63>();
    test_to_string<64>();
    test_to_string<65>();
    test_to_string<1000>();

    return 0;
}
