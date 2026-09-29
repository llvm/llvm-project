//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// [[nodiscard]] static unsigned int hardware_concurrency() noexcept;

#include <cassert>
#include <concepts>
#include <thread>
#include <type_traits>

#include "test_macros.h"

static_assert(noexcept(std::jthread::hardware_concurrency()));

int main(int, char**) {
  std::same_as<unsigned int> decltype(auto) result = std::jthread::hardware_concurrency();
  assert(result == std::thread::hardware_concurrency());

  return 0;
}
