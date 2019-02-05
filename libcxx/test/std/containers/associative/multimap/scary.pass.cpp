//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class map class multimap

// Extension:  SCARY/N2913 iterator compatibility between map and multimap

#include <map>

int main()
{
    typedef std::map<int, int> M1;
    typedef std::multimap<int, int> M2;
    M2::iterator i;
    M1::iterator j = i;
    ((void)j);
}
