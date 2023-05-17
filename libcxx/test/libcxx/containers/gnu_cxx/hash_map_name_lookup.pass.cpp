//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: modules-build

// Poison the std:: names we might use inside __gnu_cxx to ensure they're
// properly qualified.
struct allocator;
struct pair;
struct equal_to;
struct unique_ptr;

// Prevent <ext/hash_map> from generating deprecated warnings for this test.
// ADDITIONAL_COMPILE_FLAGS: -Wno-deprecated

#include <ext/hash_map>

#include "test_macros.h"


namespace __gnu_cxx {
template class hash_map<int, int>;
}

int main(int, char**) {
  typedef __gnu_cxx::hash_map<int, int> Map;
  Map m;
  Map m2(m);
  ((void)m2);

  return 0;
}
