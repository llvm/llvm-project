//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <map>

// class map

// mapped_type& operator[](const key_type& k);

// https://bugs.llvm.org/show_bug.cgi?id=16542

#include <map>


#include <tuple>


int main(int, char**)
{
    using namespace std;
    map<tuple<int,int>, size_t> m;
    m[make_tuple(2,3)]=7;

  return 0;
}
