//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <class C> auto end(const C& c) -> decltype(c.end());

#include <vector>
#include <cassert>

int main(int, char**)
{
    int ia[] = {1, 2, 3};
    const std::vector<int> v(ia, ia + sizeof(ia)/sizeof(ia[0]));
    std::vector<int>::const_iterator i = end(v);
    assert(i == v.cend());

  return 0;
}
