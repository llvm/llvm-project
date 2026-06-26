//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <map>

// class map

//    template<class K>
//        constexpr node_type extract(K&& k) const;        // C++23

#include <map>
#include "test_transparent_associative.h"

TEST_CONSTEXPR_CXX26 bool test() {
  test_transparent_extract<std::map<StoredType<int>, int, transparent_comparator_base>>({{1, 0}, {2, 0}, {4, 0}});

  test_transparent_extract<std::map<StoredType<int>, int, transparent_comparator_final>>({{1, 0}, {2, 0}, {4, 0}});

  test_non_transparent_extract<std::map<StoredType<int>, int, std::less<StoredType<int>>>>({{1, 0}, {2, 0}});

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
