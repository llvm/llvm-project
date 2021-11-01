//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>
// UNSUPPORTED: c++03, c++11, c++14

// template<class InputIterator,
//      class Allocator = allocator<typename iterator_traits<InputIterator>::value_type>>
//  basic_string(InputIterator, InputIterator, Allocator = Allocator())
//    -> basic_string<typename iterator_traits<InputIterator>::value_type,
//                 char_traits<typename iterator_traits<InputIterator>::value_type>,
//                 Allocator>;
//
//  The deduction guide shall not participate in overload resolution if InputIterator
//  is a type that does not qualify as an input iterator, or if Allocator is a type
//  that does not qualify as an allocator.


#include <string>
#include <iterator>
#include <cassert>
#include <cstddef>

#include "test_macros.h"

class NotAnIterator {};

template <typename T>
struct NotAnAllocator { typedef T value_type; };

int main(int, char**)
{
    { // Not an iterator at all
    std::basic_string s1{NotAnIterator{}, NotAnIterator{}, std::allocator<char>{}}; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'basic_string'}}
    }
    { // Not an input iterator
    std::basic_string<char16_t> s0;
    std::basic_string s1{std::back_insert_iterator(s0), //  expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'basic_string'}}
                         std::back_insert_iterator(s0),
                         std::allocator<char16_t>{}};
    }
    { // Not an allocator
    const wchar_t* s = L"12345678901234";
    (void)s;
    std::basic_string s1{s, s+10, NotAnAllocator<wchar_t>{}}; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'basic_string'}}
    }


  return 0;
}
