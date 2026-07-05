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

// test for explicit

#include <iterator>
#include <vector>

int main(int, char**)
{
    std::vector<int> v;
    std::back_insert_iterator<std::vector<int> > i = v;

  return 0;
}
