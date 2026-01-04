//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated

// hash_multimap::hash_multimap(const hash_multimap&)

#include <cassert>
#include <ext/hash_map>

int main(int, char**) {
  __gnu_cxx::hash_multimap<int, int> map;

  map.insert(std::make_pair(1, 1));
  map.insert(std::make_pair(1, 1));

  auto map2 = map;

  assert(map2.size() == 2);

  return 0;
}
