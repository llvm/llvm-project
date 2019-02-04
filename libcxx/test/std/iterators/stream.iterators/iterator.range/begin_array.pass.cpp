//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <class T, size_t N> T* begin(T (&array)[N]);

#include <iterator>
#include <cassert>

int main()
{
    int ia[] = {1, 2, 3};
    int* i = std::begin(ia);
    assert(*i == 1);
    *i = 2;
    assert(ia[0] == 2);
}
