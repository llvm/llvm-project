//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <memory>

// Test that <memory> provides all of the arithmetic, enum, and pointer
// hash specializations.

#include <memory>

#include "poisoned_hash_helper.hpp"

int main() {
  test_library_hash_specializations_available();
}
