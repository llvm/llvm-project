//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <unordered_set>

// class unordered_multiset

// template <class K>
// node_type extract(K&& k);

#include <unordered_set>

#include "test_transparent_unordered.h"

int main(int, char**) {
  using key_type = StoredType<int>;

  {
    // Make sure conversions don't happen for transparent non-final hasher and key_equal
    using S = unord_set_type<std::unordered_multiset, transparent_hash, std::equal_to<>>;
    test_transparent_extract<S>({1, 2, 4});
  }

  {
    // Make sure conversions don't happen for transparent final hasher and key_equal
    using S = unord_set_type<std::unordered_multiset, transparent_hash_final, transparent_equal_final>;
    test_transparent_extract<S>({1, 2, 4});
  }

  {
    // Make sure conversions do happen for non-transparent hasher
    using S = unord_set_type<std::unordered_multiset, non_transparent_hash, std::equal_to<>>;
    test_non_transparent_extract<S>({1, 2});
  }

  {
    // Make sure conversions do happen for non-transparent key_equal
    using S = unord_set_type<std::unordered_multiset, transparent_hash, std::equal_to<key_type>>;
    test_non_transparent_extract<S>({1, 2});
  }

  {
    // Make sure conversions do happen for both non-transparent hasher and key_equal
    using S = unord_set_type<std::unordered_multiset, non_transparent_hash, std::equal_to<key_type>>;
    test_non_transparent_extract<S>({1, 2});
  }

  return 0;
}
