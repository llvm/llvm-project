//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// class ostream_iterator

// ostream_iterator& operator++();
// ostream_iterator& operator++(int);

#include <iterator>
#include <sstream>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::ostringstream os;
    std::ostream_iterator<int> i(os);
    std::ostream_iterator<int>& iref1 = ++i;
    assert(&iref1 == &i);
    std::ostream_iterator<int>& iref2 = i++;
    assert(&iref2 == &i);

  return 0;
}
