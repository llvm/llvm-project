//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_map

// mapped_type& operator[](const key_type& k);

// https://llvm.org/PR16542

#include <cstddef>
#include <tuple>
#include <unordered_map>

struct my_hash {
    std::size_t operator()(const std::tuple<int, int>&) const { return 0; }
};

int main(int, char**) {
    std::unordered_map<std::tuple<int, int>, std::size_t, my_hash> m;
    m[std::make_tuple(2, 3)] = 7;

    return 0;
}
