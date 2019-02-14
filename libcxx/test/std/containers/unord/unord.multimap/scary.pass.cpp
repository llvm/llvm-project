//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// class unordered_map class unordered_multimap

// Extension:  SCARY/N2913 iterator compatibility between unordered_map and unordered_multimap

#include <unordered_map>

int main(int, char**)
{
    typedef std::unordered_map<int, int> M1;
    typedef std::unordered_multimap<int, int> M2;
    M2::iterator i;
    M1::iterator j = i;
    ((void)j);

  return 0;
}
