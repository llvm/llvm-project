//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// back_insert_iterator

// explicit back_insert_iterator(Cont& x);

#include <iterator>
#include <vector>
#include "nasty_containers.hpp"

template <class C>
void
test(C c)
{
    std::back_insert_iterator<C> i(c);
}

int main(int, char**)
{
    test(std::vector<int>());
    test(nasty_vector<int>());

  return 0;
}
