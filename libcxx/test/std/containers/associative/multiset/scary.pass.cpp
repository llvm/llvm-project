//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set class multiset

// Extension:  SCARY/N2913 iterator compatibility between set and multiset

#include <set>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::set<int> M1;
    typedef std::multiset<int> M2;
    M2::iterator i;
    M1::iterator j = i;
    ((void)j);

  return 0;
}
