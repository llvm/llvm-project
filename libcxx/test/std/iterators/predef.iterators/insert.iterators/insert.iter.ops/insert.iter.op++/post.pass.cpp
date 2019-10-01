//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// insert_iterator

// insert_iterator<Cont> operator++(int);

#include <iterator>
#include <vector>
#include <cassert>
#include "nasty_containers.hpp"

#include "test_macros.h"

template <class C>
void
test(C c)
{
    std::insert_iterator<C> i(c, c.end());
    std::insert_iterator<C> r = i++;
    r = 0;
    assert(c.size() == 1);
    assert(c.back() == 0);
}

int main(int, char**)
{
    test(std::vector<int>());
    test(nasty_vector<int>());

  return 0;
}
