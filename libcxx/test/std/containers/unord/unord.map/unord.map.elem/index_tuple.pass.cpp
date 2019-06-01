//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_map

// mapped_type& operator[](const key_type& k);

// https://bugs.llvm.org/show_bug.cgi?id=16542

#include <unordered_map>
#include <tuple>

#include "test_macros.h"

using namespace std;

struct my_hash
{
    size_t operator()(const tuple<int,int>&) const {return 0;}
};

int main(int, char**)
{
    unordered_map<tuple<int,int>, size_t, my_hash> m;
    m[make_tuple(2,3)]=7;

  return 0;
}
