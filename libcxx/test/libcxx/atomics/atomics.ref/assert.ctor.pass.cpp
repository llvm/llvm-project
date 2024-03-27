//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-hardening-mode=none || libcpp-hardening-mode=fast
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

// <atomic>

// atomic_ref(T& obj);
//
// Preconditions: The referenced object is aligned to required_alignment.

#include <atomic>

#include "check_assertion.h"

int main(int, char**) {
  {
    char c[8];
    float* f = new (c) float(3.14f);
    [[maybe_unused]] std::atomic_ref<float> r(*f);
  }

  TEST_LIBCPP_ASSERT_FAILURE(
      ([] {
        char c[8];
        float* f = new (c + 1) float(3.14f);
        [[maybe_unused]] std::atomic_ref<float> r(*f);
      }()),
      "atomic_ref ctor: referenced object must be aligned to required_alignment");

  return 0;
}
