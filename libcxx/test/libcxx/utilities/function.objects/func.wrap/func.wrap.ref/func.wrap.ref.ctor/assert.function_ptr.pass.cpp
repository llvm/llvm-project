//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-hardening-mode=none || libcpp-hardening-mode=fast
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// template<class F> function_ref(F* f) noexcept;
// Preconditions: f is not a null pointer.

#include <functional>

#include "check_assertion.h"

int main(int, char**) {
  TEST_LIBCPP_ASSERT_FAILURE(
      std::function_ref<void()>(static_cast<void (*)()>(nullptr)), "the function pointer should not be a nullptr");

  return 0;
}
