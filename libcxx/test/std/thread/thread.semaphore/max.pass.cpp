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

// <semaphore>

#include <cstddef>
#include <semaphore>

int main(int, char**)
{
  static_assert(std::counting_semaphore<>::max() >= 1, "");
  static_assert(std::counting_semaphore<1>::max() >= 1, "");
  static_assert(std::counting_semaphore<std::numeric_limits<int>::max()>::max() >= std::numeric_limits<int>::max(), "");
  static_assert(std::counting_semaphore<std::numeric_limits<std::ptrdiff_t>::max()>::max() == std::numeric_limits<ptrdiff_t>::max(), "");
  return 0;
}
