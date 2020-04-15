//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// GCC 5 does not evaluate static assertions dependent on a template parameter.
// UNSUPPORTED: gcc-5

// <iterator>

// reverse_iterator

// explicit reverse_iterator(Iter x);

// test explicit

#include <iterator>

template <class It>
void
test(It i)
{
    std::reverse_iterator<It> r = i;
}

int main(int, char**)
{
    const char s[] = "123";
    test(s);

  return 0;
}
