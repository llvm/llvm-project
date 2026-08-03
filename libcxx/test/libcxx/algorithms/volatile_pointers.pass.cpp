//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Algorithms in libc++ may make alignment and dereferenceability assumptions
// for optimization purposes. This test ensures that we don't emit these assumptions
// for volatile pointers, as doing so can break algorithms operating on volatile data.

// UNSUPPORTED: c++03

#include <__memory/valid_range.h>
#include <algorithm>
#include <cassert>

#include "test_macros.h"

TEST_CONSTEXPR_CXX14 bool test_constexpr() {
  int arr[3] = {1, 2, 3};
  std::__assume_valid_range(arr, arr + 3);
  return true;
}

void test_volatile_pointers() {
  volatile int arr[3] = {1, 1, 1};
  std::__assume_valid_range(arr, arr + 3);

  bool all_ones = std::all_of(arr, arr + 3, [](volatile int& val) { return val == 1; });
  assert(all_ones);

  auto* found = std::find_if(arr, arr + 3, [](volatile int& val) { return val == 1; });
  assert(found == arr);
}

int main(int, char**) {
  test_constexpr();
  test_volatile_pointers();

#if TEST_STD_VER >= 14
  static_assert(test_constexpr(), "");
#endif

  return 0;
}
