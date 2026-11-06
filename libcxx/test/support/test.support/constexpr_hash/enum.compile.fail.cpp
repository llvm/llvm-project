//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// "support/constexpr_hash.h"

//  Hashing a struct w/o a defined hash should fail.

#include "constexpr_hash.h"

#include <cstddef>

struct X {};

int main(int, char**) {
  X x;
  std::size_t h = support::constexpr_hash<X>{}(x);

  return 0;
}
