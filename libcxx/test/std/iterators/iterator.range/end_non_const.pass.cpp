//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <class C> auto end(C& c) -> decltype(c.end());

#include <vector>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    int ia[] = {1, 2, 3};
    std::vector<int> v(ia, ia + sizeof(ia)/sizeof(ia[0]));
    std::vector<int>::iterator i = end(v);
    assert(i == v.end());

  return 0;
}
