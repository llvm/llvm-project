//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <class C> auto begin(C& c) -> decltype(c.begin());

#include <vector>
#include <cassert>

int main(int, char**)
{
    int ia[] = {1, 2, 3};
    std::vector<int> v(ia, ia + sizeof(ia)/sizeof(ia[0]));
    std::vector<int>::iterator i = begin(v);
    assert(*i == 1);
    *i = 2;
    assert(*i == 2);

  return 0;
}
