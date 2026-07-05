//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class istream_iterator

// istream_iterator(istream_type& s);

#include <iterator>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::istringstream inf(" 1 23");
    std::istream_iterator<int> i(inf);
    assert(i != std::istream_iterator<int>());
    assert(inf.peek() == ' ');
    assert(inf.good());
    int j = 0;
    inf >> j;
    assert(j == 23);

  return 0;
}
