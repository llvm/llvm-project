//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-threads
// XFAIL: availability-hazard_pointer-missing

// <hazard_pointer>

// hazard_pointer() noexcept;   Postconditions: *this is empty.

#include <hazard_pointer>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  std::hazard_pointer h;
  assert(h.empty());
  const std::hazard_pointer ch;
  assert(ch.empty());
  return 0;
}
