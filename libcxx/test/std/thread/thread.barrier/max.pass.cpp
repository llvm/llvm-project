//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11

// <barrier>

#include <barrier>
#include <thread>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(std::barrier<>::max() > 0, "");
  auto l = [](){};
  static_assert(std::barrier<decltype(l)>::max() > 0, "");
  return 0;
}
