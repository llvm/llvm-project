//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <map>

// class multimap

//    template<class K>
//        size_type erase(K&& k) const;        // C++23

#include <map>
#include "test_transparent_associative.h"

int main(int, char**) {
  test_transparent_erase<std::multimap<StoredType<int>, int, transparent_comparator_base>>(
      {{1, 0}, {2, 0}, {4, 0}, {5, 0}});

  test_transparent_erase<std::multimap<StoredType<int>, int, transparent_comparator_final>>(
      {{1, 0}, {2, 0}, {4, 0}, {5, 0}});

  test_non_transparent_erase<std::multimap<StoredType<int>, int, std::less<StoredType<int>>>>({{1, 0}, {2, 0}});
}
