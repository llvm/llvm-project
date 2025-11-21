//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated

// hash_multimap::insert

#include <cassert>
#include <ext/hash_set>

int main(int, char**) {
  __gnu_cxx::hash_multiset<int> map;

  map.insert(1);
  map.insert(1);

  assert(map.size() == 2);
  assert(map.equal_range(1).first == map.begin());
  assert(map.equal_range(1).second == map.end());

  int arr[] = {1, 1};

  map.insert(arr, arr + 2);

  assert(map.size() == 4);
  assert(map.equal_range(1).first == map.begin());
  assert(map.equal_range(1).second == map.end());

  return 0;
}
