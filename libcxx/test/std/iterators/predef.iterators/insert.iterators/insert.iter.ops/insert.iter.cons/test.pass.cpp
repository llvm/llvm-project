//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// insert_iterator

// insert_iterator(Cont& x, Cont::iterator i);

#include <iterator>
#include <vector>
#include "nasty_containers.hpp"

template <class C>
void
test(C c)
{
    std::insert_iterator<C> i(c, c.begin());
}

int main()
{
    test(std::vector<int>());
    test(nasty_vector<int>());
}
