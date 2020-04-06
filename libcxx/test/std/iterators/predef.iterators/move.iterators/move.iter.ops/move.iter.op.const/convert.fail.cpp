//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// GCC 5 does not evaluate static assertions dependent on a template parameter.
// UNSUPPORTED: gcc-5

// <iterator>

// move_iterator

// template <class U>
//   requires HasConstructor<Iter, const U&>
//   move_iterator(const move_iterator<U> &u);

// test requires

#include <iterator>

template <class It, class U>
void
test(U u)
{
    std::move_iterator<U> r2(u);
    std::move_iterator<It> r1 = r2;
}

struct base {};
struct derived {};

int main(int, char**)
{
    derived d;

    test<base*>(&d);

  return 0;
}
