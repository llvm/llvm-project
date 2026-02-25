//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated
#include <ext/hash_map>

int main(int, char**) {
  __gnu_cxx::hash_multimap<const char*, std::string> m;
  auto it = m.insert(std::make_pair("foo", "bar"));
  return it->first == nullptr;
}
