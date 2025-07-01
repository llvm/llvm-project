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
#include <ranges>

int main() {
  bool status = std::ranges::range<std::optional<int>>;
  std::cout << "std::ranges::range<std::optional<int>> is " << status << std::endl;
  return 0;
}