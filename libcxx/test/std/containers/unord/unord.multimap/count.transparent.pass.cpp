//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_multimap

// template <typename K>
// size_type count(const K& k) const;

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <unordered_map>

#include "test_transparent_unordered.h"

int main(int, char**)
{
    using key_type = StoredType<int>;

    {
      // Make sure conversions don't happen for transparent non-final hasher and key_equal
      using M = unord_map_type<std::unordered_multimap, transparent_hash, std::equal_to<>>;
      test_transparent_count<M>({{1, 2}, {1, 3}, {2, 3}});
      test_transparent_count<const M>({{1, 2}, {1, 3}, {2, 3}});
    }

    {
      // Make sure conversions don't happen for transparent final hasher and key_equal
      using M = unord_map_type<std::unordered_multimap, transparent_hash_final, transparent_equal_final>;
      test_transparent_count<M>({{1, 2}, {1, 3}, {2, 3}});
      test_transparent_count<const M>({{1, 2}, {1, 3}, {2, 3}});
    }

    {
      // Make sure conversions do happen for non-transparent hasher
      using M = unord_map_type<std::unordered_multimap, non_transparent_hash, std::equal_to<>>;
      test_non_transparent_count<M>({{1, 2}, {1, 3}, {2, 3}});
      test_non_transparent_count<const M>({{1, 2}, {1, 3}, {2, 3}});
    }

    {
      // Make sure conversions do happen for non-transparent key_equal
      using M = unord_map_type<std::unordered_multimap, transparent_hash, std::equal_to<key_type>>;
      test_non_transparent_count<M>({{1, 2}, {1, 3}, {2, 3}});
      test_non_transparent_count<const M>({{1, 2}, {1, 3}, {2, 3}});
    }

    {
      // Make sure conversions do happen for both non-transparent hasher and key_equal
      using M = unord_map_type<std::unordered_multimap, non_transparent_hash, std::equal_to<key_type>>;
      test_non_transparent_count<M>({{1, 2}, {1, 3}, {2, 3}});
      test_non_transparent_count<const M>({{1, 2}, {1, 3}, {2, 3}});
    }

    return 0;
}
