//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Ensure that APIs which take a pointer are diagnosing passing a nullptr to them

#include <memory>

#include "test_macros.h"

void func() {
  using Arr     = int[1];
  int* const np = nullptr;

#if TEST_STD_VER >= 20
  Arr* const np2 = nullptr;
  std::construct_at(np); // expected-warning {{null passed}}
  std::destroy_at(np2);  // expected-warning {{null passed}}
#endif

  std::destroy_at(np); // expected-warning {{null passed}}
}
