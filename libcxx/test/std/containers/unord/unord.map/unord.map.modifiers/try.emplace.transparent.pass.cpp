//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14

// <unordered_map>

// class unordered_map

// template <class K, class... Args>
//  pair<iterator, bool> try_emplace(K&& k, Args&&... args);               // C++XX
// template <class K, class... Args>
//  pair<iterator, bool> try_emplace(K&& k, Args&&... args);               // C++XX
// template <class K, class... Args>
//  iterator try_emplace(const_iterator hint, K&& k, Args&&... args);      // C++XX
// template <class K, class... Args>
//  iterator try_emplace(const_iterator hint, K&& k, Args&&... args);      // C++XX

#include <unordered_map>

#include "test_transparent_unordered.h"

int main(int, char**) {
    using key_type = StoredType2<int>;

    {
        using map_type = unord_map_type2<std::unordered_map, transparent_hash,
                                         std::equal_to<> >;
        test_transparent_try_emplace<map_type>();
    }

    {
        using map_type = unord_map_type2<std::unordered_map, transparent_hash_final,
                                         transparent_equal_final>;
        test_transparent_try_emplace<map_type>();
    }

    {
        using map_type = unord_map_type2<std::unordered_map, non_transparent_hash,
                                         std::equal_to<> >;
        test_non_transparent_try_emplace<map_type>();
    }

    {
        using map_type = unord_map_type2<std::unordered_map, transparent_hash,
                                         std::equal_to<key_type> >;
        test_non_transparent_try_emplace<map_type>();
    }

    {
        using map_type = unord_map_type2<std::unordered_map, non_transparent_hash,
                                         std::equal_to<key_type> >;
        test_non_transparent_try_emplace<map_type>();
    }
}
