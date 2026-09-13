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

// bool empty() const noexcept;

#include <hazard_pointer>
#include <cassert>
#include <utility>

#include "test_macros.h"

int main(int, char**) {
  const std::hazard_pointer e;
  assert(e.empty());
  std::hazard_pointer h = std::make_hazard_pointer();
  assert(!h.empty());
  std::hazard_pointer m = std::move(h);
  assert(h.empty());
  assert(!m.empty());
  return 0;
}
