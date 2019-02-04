//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>
// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides


// template <class InputIterator, class Allocator = allocator<typename iterator_traits<InputIterator>::value_type>>
//    forward_list(InputIterator, InputIterator, Allocator = Allocator())
//    -> forward_list<typename iterator_traits<InputIterator>::value_type, Allocator>;
//


#include <forward_list>
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
//  forward_list (allocator &)
    std::forward_list fwl((std::allocator<int>()));  // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'forward_list'}}
//  Note: The extra parens are necessary, since otherwise clang decides it is a function declaration.
//  Also, we can't use {} instead of parens, because that constructs a
//      forward_list<allocator<int>, allocator<allocator<int>>>
    }


  return 0;
}
