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

// XFAIL: availability-synchronization_library-missing

// <latch>

#include <latch>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
  std::latch l(1);

  l.count_down();
  bool const b = l.try_wait();
  assert(b);

  return 0;
}
