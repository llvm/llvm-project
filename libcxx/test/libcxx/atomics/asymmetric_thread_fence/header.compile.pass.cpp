//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++11
// UNSUPPORTED: no-threads

// <__atomic/asymmetric_thread_fence.h> is self-contained and warning-free in every language mode
// (the test suite compiles with -Wunused-parameter -Werror and _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER,
// so an unused memory_order parameter without [[maybe_unused]] fails here).

#include <__atomic/asymmetric_thread_fence.h>
#include <__atomic/memory_order.h>

void test(std::memory_order order) {
  static_assert(noexcept(std::__asymmetric_thread_fence_light(order)), "");
  static_assert(noexcept(std::__asymmetric_thread_fence_heavy(order)), "");
  std::__asymmetric_thread_fence_light(order);
  std::__asymmetric_thread_fence_heavy(order);
}
