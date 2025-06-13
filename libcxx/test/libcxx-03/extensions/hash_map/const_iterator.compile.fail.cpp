//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ext/hash_map>

int main(int, char**)
{
    __gnu_cxx::hash_map<int, int> m;
    m[1] = 1;
    const __gnu_cxx::hash_map<int, int> &cm = m;
    cm.find(1)->second = 2;  // error

  return 0;
}
