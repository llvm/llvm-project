//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// "support/constexpr_hash.h"

#include "constexpr_hash.h"

#include <cstddef>

int main(int, char**) {

  typedef int* T;
  using H = support::constexpr_hash<T>;

  std::size_t h = (H()(T()));

  return 0;
}
