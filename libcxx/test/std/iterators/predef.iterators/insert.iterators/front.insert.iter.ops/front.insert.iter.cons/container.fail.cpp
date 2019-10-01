//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// front_insert_iterator

// explicit front_insert_iterator(Cont& x);

// test for explicit

#include <iterator>
#include <list>

int main(int, char**)
{
    std::front_insert_iterator<std::list<int> > i = std::list<int>();

  return 0;
}
