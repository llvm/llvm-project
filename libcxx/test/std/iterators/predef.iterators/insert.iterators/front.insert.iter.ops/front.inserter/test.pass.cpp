//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <BackInsertionContainer Cont>
//   front_insert_iterator<Cont>
//   front_inserter(Cont& x);

#include <iterator>
#include <list>
#include <cassert>
#include "nasty_containers.hpp"

template <class C>
void
test(C c)
{
    std::front_insert_iterator<C> i = std::front_inserter(c);
    i = 0;
    assert(c.size() == 1);
    assert(c.front() == 0);
}

int main(int, char**)
{
    test(std::list<int>());
    test(nasty_list<int>());

  return 0;
}
