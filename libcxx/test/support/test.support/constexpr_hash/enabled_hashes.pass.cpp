//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// "support/constexpr_hash.h"

// Test that "support/constexpr_hash.h" provides all of the arithmetic, enum, and pointer
// hash specializations, and that they work during constant evaluation.

#include "constexpr_hash.h"

#include "poisoned_hash_helper.h"

#include "test_macros.h"

int main(int, char**) {
  test_library_hash_specializations_available();

  static_assert(test_constexpr_hash_specializations_available());

  return 0;
}
