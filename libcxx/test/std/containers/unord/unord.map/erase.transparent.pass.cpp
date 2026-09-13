//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <unordered_map>

// class unordered_map

// template <class K>
// size_type erase(K&& k);

#include <unordered_map>

#include "test_transparent_unordered.h"

int main(int, char**) {
  using key_type = StoredType<int>;

  {
    // Make sure conversions don't happen for transparent non-final hasher and key_equal
    using M = unord_map_type<std::unordered_map, transparent_hash, std::equal_to<>>;
    test_transparent_erase<M>({{1, 0}, {2, 0}, {4, 0}, {5, 0}});
  }

  {
    // Make sure conversions don't happen for transparent final hasher and key_equal
    using M = unord_map_type<std::unordered_map, transparent_hash_final, transparent_equal_final>;
    test_transparent_erase<M>({{1, 0}, {2, 0}, {4, 0}, {5, 0}});
  }

  {
    // Make sure conversions do happen for non-transparent hasher
    using M = unord_map_type<std::unordered_map, non_transparent_hash, std::equal_to<>>;
    test_non_transparent_erase<M>({{1, 0}, {2, 0}});
  }

  {
    // Make sure conversions do happen for non-transparent key_equal
    using M = unord_map_type<std::unordered_map, transparent_hash, std::equal_to<key_type>>;
    test_non_transparent_erase<M>({{1, 0}, {2, 0}});
  }

  {
    // Make sure conversions do happen for both non-transparent hasher and key_equal
    using M = unord_map_type<std::unordered_map, non_transparent_hash, std::equal_to<key_type>>;
    test_non_transparent_erase<M>({{1, 0}, {2, 0}});
  }

  return 0;
}
