//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>
// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides


// template <class InputIterator, class Allocator = allocator<typename iterator_traits<InputIterator>::value_type>>
//    deque(InputIterator, InputIterator, Allocator = Allocator())
//    -> deque<typename iterator_traits<InputIterator>::value_type, Allocator>;
//


#include <deque>
#include <iterator>
#include <cassert>
#include <cstddef>
#include <climits> // INT_MAX

struct A {};

int main(int, char**)
{
//  Test the explicit deduction guides

//  Test the implicit deduction guides
    {
//  deque (allocator &)
    std::deque deq((std::allocator<int>()));  // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'deque'}}
//  Note: The extra parens are necessary, since otherwise clang decides it is a function declaration.
//  Also, we can't use {} instead of parens, because that constructs a
//      deque<allocator<int>, allocator<allocator<int>>>
    }


  return 0;
}
