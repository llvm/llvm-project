//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://llvm.org/PR40995 is fixed
// UNSUPPORTED: availability-pmr-missing

// <memory_resource>

// class monotonic_buffer_resource

#include <memory_resource>
#include <cassert>

#include "count_new.h"

int main(int, char**) {
  globalMemCounter.reset();
  std::pmr::monotonic_buffer_resource mono;

  for (int i = 0; i < 100; ++i) {
    (void)mono.allocate(1);
    assert(globalMemCounter.last_new_size < 1000000000);
    mono.release();
    assert(globalMemCounter.checkOutstandingNewEq(0));
  }

  return 0;
}
