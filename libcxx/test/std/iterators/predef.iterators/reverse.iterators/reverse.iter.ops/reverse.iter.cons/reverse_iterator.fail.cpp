//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// template <class U>
//   requires HasConstructor<Iter, const U&>
//   reverse_iterator(const reverse_iterator<U> &u);

// test requires

#include <iterator>

template <class It, class U>
void
test(U u)
{
    std::reverse_iterator<U> r2(u);
    std::reverse_iterator<It> r1 = r2;
}

struct base {};
struct derived {};

int main()
{
    derived d;

    test<base*>(&d);
}
