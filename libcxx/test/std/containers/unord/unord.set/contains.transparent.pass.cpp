//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_set

// template <typename K>
// bool contains(const K& x) const;

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <unordered_set>
#include "test_transparent_unordered.h"

int main(int, char**)
{
    using key_type = StoredType<int>;

    {
      // Make sure conversions don't happen for transparent non-final hasher and key_equal
      using S = unord_set_type<std::unordered_set, transparent_hash, std::equal_to<>>;
      test_transparent_contains<S>({1, 2});
      test_transparent_contains<const S>({1, 2});
    }

    {
      // Make sure conversions don't happen for transparent final hasher and key_equal
      using S = unord_set_type<std::unordered_set, transparent_hash_final, transparent_equal_final>;
      test_transparent_contains<S>({1, 2});
      test_transparent_contains<const S>({1, 2});
    }

    {
      // Make sure conversions do happen for non-transparent hasher
      using S = unord_set_type<std::unordered_set, non_transparent_hash, std::equal_to<>>;
      test_non_transparent_contains<S>({1, 2});
      test_non_transparent_contains<const S>({1, 2});
    }

    {
      // Make sure conversions do happen for non-transparent key_equal
      using S = unord_set_type<std::unordered_set, transparent_hash, std::equal_to<key_type>>;
      test_non_transparent_contains<S>({1, 2});
      test_non_transparent_contains<const S>({1, 2});
    }

    {
      // Make sure conversions do happen for both non-transparent hasher and key_equal
      using S = unord_set_type<std::unordered_set, non_transparent_hash, std::equal_to<key_type>>;
      test_non_transparent_contains<S>({1, 2});
      test_non_transparent_contains<const S>({1, 2});
    }

    return 0;
}
