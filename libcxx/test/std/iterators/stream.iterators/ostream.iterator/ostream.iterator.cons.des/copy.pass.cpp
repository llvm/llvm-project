//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class ostream_iterator

// ostream_iterator(const ostream_iterator& x);

#include <iterator>
#include <sstream>
#include <cassert>

int main()
{
    std::ostringstream outf;
    std::ostream_iterator<int> i(outf);
    std::ostream_iterator<int> j = i;
    assert(outf.good());
    ((void)j);
}
