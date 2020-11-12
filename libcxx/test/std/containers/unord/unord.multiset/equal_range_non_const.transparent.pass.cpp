//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_multiset

// template <typename K>
// pair<iterator, iterator> equal_range(const K& k);

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <unordered_set>

#include "test_transparent_unordered.h"

int main(int, char**)
{
    using key_type = StoredType<int>;

    {
      // Make sure conversions don't happen for transparent non-final hasher and key_equal
      using set_type = unord_set_type<std::unordered_multiset, transparent_hash,
                                 std::equal_to<> >;
      test_transparent_equal_range<set_type>(key_type{1}, key_type{1}, key_type{2});
    }

    {
      // Make sure conversions don't happen for transparent final hasher and key_equal
      using set_type = unord_set_type<std::unordered_multiset, transparent_hash_final,
                                 transparent_equal_final>;
      test_transparent_equal_range<set_type>(key_type{1}, key_type{1}, key_type{2});
    }

    {
      // Make sure conversions do happen for non-transparent hasher
      using set_type = unord_set_type<std::unordered_multiset, non_transparent_hash,
                                 std::equal_to<> >;
      test_non_transparent_equal_range<set_type>(key_type{1}, key_type{1}, key_type{2});
    }

    {
      // Make sure conversions do happen for non-transparent key_equal
      using set_type = unord_set_type<std::unordered_multiset, transparent_hash,
                                 std::equal_to<key_type> >;
      test_non_transparent_equal_range<set_type>(key_type{1}, key_type{1}, key_type{2});
    }

    {
      // Make sure conversions do happen for both non-transparent hasher and key_equal
      using set_type = unord_set_type<std::unordered_multiset, non_transparent_hash,
                                 std::equal_to<key_type> >;
      test_non_transparent_equal_range<set_type>(key_type{1}, key_type{1}, key_type{2});
    }

    return 0;
}
