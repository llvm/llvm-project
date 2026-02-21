//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <set>

// class multiset

//    template<class K>
//        size_type erase(K&& k) const;        // C++23

#include <set>
#include "test_transparent_associative.h"

int main(int, char**) {
  test_transparent_erase<std::multiset<StoredType<int>, transparent_comparator_base>>({1, 2, 4, 5});

  test_transparent_erase<std::multiset<StoredType<int>, transparent_comparator_final>>({1, 2, 4, 5});

  test_non_transparent_erase<std::multiset<StoredType<int>, std::less<StoredType<int>>>>({1, 2});
}
