//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>
// UNSUPPORTED: c++03, c++11, c++14

#include <queue>
#include <list>
#include <iterator>
#include <cassert>
#include <cstddef>


int main(int, char**)
{
//  Test the explicit deduction guides
    {
//  queue(const Container&, const Alloc&);
//  The '45' is not an allocator
    std::queue que(std::list<int>{1,2,3}, 45);  // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'queue'}}
    }

    {
//  queue(const queue&, const Alloc&);
//  The '45' is not an allocator
    std::queue<int> source;
    std::queue que(source, 45);  // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'queue'}}
    }

//  Test the implicit deduction guides
    {
//  queue (allocator &)
    std::queue que((std::allocator<int>()));  // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'queue'}}
//  Note: The extra parens are necessary, since otherwise clang decides it is a function declaration.
//  Also, we can't use {} instead of parens, because that constructs a
//      stack<allocator<int>, allocator<allocator<int>>>
    }


  return 0;
}
