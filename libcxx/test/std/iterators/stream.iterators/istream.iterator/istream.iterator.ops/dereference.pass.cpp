//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class istream_iterator

// const T& operator*() const;

#include <iterator>
#include <sstream>
#include <cassert>

int main()
{
    std::istringstream inf(" 1 23");
    std::istream_iterator<int> i(inf);
    int j = 0;
    j = *i;
    assert(j == 1);
    j = *i;
    assert(j == 1);
    ++i;
    j = *i;
    assert(j == 23);
    j = *i;
    assert(j == 23);
}
