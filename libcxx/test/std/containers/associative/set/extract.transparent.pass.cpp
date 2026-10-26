//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <set>

// class set

//    template<class K>
//        constexpr node_type extract(K&& k) const;        // C++23, constexpr since C++26

#include <set>
#include "test_transparent_associative.h"

TEST_CONSTEXPR_CXX26 bool test() {
  test_transparent_extract<std::set<StoredType<int>, transparent_comparator_base>>({1, 2, 4});

  test_transparent_extract<std::set<StoredType<int>, transparent_comparator_final>>({1, 2, 4});

  test_non_transparent_extract<std::set<StoredType<int>, std::less<StoredType<int>>>>({1, 2});

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
