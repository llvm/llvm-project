//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// The default allocator must be usable with only <ranges>, including with modules enabled.
#include <ranges>

void test() {
  int range[]                     = {1};
  auto elements                   = std::ranges::elements_of(range);
  [[maybe_unused]] auto allocator = elements.allocator;
}
