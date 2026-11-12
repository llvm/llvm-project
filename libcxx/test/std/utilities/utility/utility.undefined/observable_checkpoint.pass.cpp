//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// REQUIRES: has-observable-checkpoint

// void observable_checkpoint() noexcept;

#include <type_traits>
#include <utility>

#include "test_macros.h"

int main(int, char**) {
  static_assert(std::is_same_v<decltype(std::observable_checkpoint()), void>);
  static_assert(noexcept(std::observable_checkpoint()));
  static_assert(std::is_same_v<decltype(&std::observable_checkpoint), void (*)() noexcept>);

  std::observable_checkpoint();
  auto checkpoint = &std::observable_checkpoint;
  checkpoint();

  return 0;
}
