//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: std-at-least-c++26

#include <iostream>
#include <optional>

int main() {
  std::optional<int> val = 2;
  auto end               = val.end();
  auto s                 = *end;
  (void)s;
  return 0;
}