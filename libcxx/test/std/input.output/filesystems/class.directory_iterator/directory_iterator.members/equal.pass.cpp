//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem

// <filesystem>

// class directory_iterator

// bool operator==(default_sentinel_t) const noexcept {           // since C++20
//   return *this == directory_iterator();
// }

#include <cassert>
#include <iterator>

#include <filesystem>
#include "test_comparisons.h"
namespace fs = std::filesystem;

int main(int, char**) {
  AssertEqualityAreNoexcept<fs::directory_iterator>();
  AssertEqualityReturnBool<fs::directory_iterator>();

  fs::directory_iterator i;
  assert(testEquality(i, std::default_sentinel, true));

  return 0;
}
