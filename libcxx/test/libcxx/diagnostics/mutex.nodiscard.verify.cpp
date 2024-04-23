//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// UNSUPPORTED: no-threads

// check that <mutex> functions are marked [[nodiscard]]

// clang-format off

#include <mutex>

#include "test_macros.h"

void test() {
  std::mutex mutex;
  std::lock_guard<std::mutex>{mutex};                  // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  std::lock_guard<std::mutex>{mutex, std::adopt_lock}; // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
}
