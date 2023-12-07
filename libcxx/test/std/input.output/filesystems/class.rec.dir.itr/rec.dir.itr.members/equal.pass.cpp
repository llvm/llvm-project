//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// class recursive_directory_iterator

// bool operator==(default_sentinel_t) const noexcept {           // since C++20
//   return *this == recursive_directory_iterator();
// }

#include <cassert>
#include <iterator>

#include "filesystem_include.h"
#include "test_comparisons.h"

int main(int, char**) {
  AssertEqualityAreNoexcept<fs::recursive_directory_iterator>();
  AssertEqualityReturnBool<fs::recursive_directory_iterator>();

  fs::recursive_directory_iterator i;
  assert(testEquality(i, std::default_sentinel, true));

  return 0;
}
