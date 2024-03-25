//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// Test that <filesystem> provides all of the arithmetic, enum, and pointer
// hash specializations.

#include <filesystem>
#include "poisoned_hash_helper.h"
namespace fs = std::filesystem;

int main(int, char**) {
  test_library_hash_specializations_available();
  test_hash_enabled_for_type<fs::path>();

  return 0;
}
