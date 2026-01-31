//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated

// hash_set::hash_set(const hash_set&)

#include <cassert>
#include <ext/hash_set>

int main(int, char**) {
  __gnu_cxx::hash_set<int> set;

  set.insert(1);
  set.insert(2);

  auto set2 = set;

  assert(set2.size() == 2);

  return 0;
}
