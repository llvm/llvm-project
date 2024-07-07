//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <utility>

// inline constexpr ignore-type ignore;

#include <utility>

#include "test_macros.h"

int main(int, char**) {
  { [[maybe_unused]] TEST_CONSTEXPR_CXX14 auto& ignore_v = std::ignore; }
}
