//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <bitset>

// Test that <bitset> provides all of the arithmetic, enum, and pointer
// hash specializations.

#include <bitset>

#include "poisoned_hash_helper.h"

#include "test_macros.h"

int main(int, char**) {
  test_library_hash_specializations_available();
  {
    test_hash_enabled_for_type<std::bitset<0> >();
    test_hash_enabled_for_type<std::bitset<1> >();
    test_hash_enabled_for_type<std::bitset<1024> >();
    test_hash_enabled_for_type<std::bitset<100000> >();
  }

  return 0;
}
