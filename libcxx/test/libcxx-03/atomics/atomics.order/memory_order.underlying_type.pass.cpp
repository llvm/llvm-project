//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that std::memory_order has the same size under all
// standard versions to make sure we're not breaking the ABI. This is
// relevant because std::memory_order is a scoped enumeration in C++20,
// but an unscoped enumeration pre-C++20.
//
// See PR40977 for details.

#include <atomic>
#include <type_traits>

#include "test_macros.h"


enum cpp17_memory_order {
  cpp17_memory_order_relaxed, cpp17_memory_order_consume, cpp17_memory_order_acquire,
  cpp17_memory_order_release, cpp17_memory_order_acq_rel, cpp17_memory_order_seq_cst
};

static_assert((std::is_same<std::underlying_type<cpp17_memory_order>::type,
                            std::underlying_type<std::memory_order>::type>::value),
  "std::memory_order should have the same underlying type as a corresponding "
  "unscoped enumeration would. Otherwise, our ABI changes from C++17 to C++20.");

int main(int, char**) {
  return 0;
}
