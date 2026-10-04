//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <rcu>

// UNSUPPORTED: no-threads
// XFAIL: libcpp-has-no-experimental-rcu
// XFAIL: availability-rcu-missing
// REQUIRES: can-test-hardening-assertions-extensive
// REQUIRES: std-at-least-c++26

#include <rcu>
#include <thread>

#include "check_assertion.h"

int main(int, char**) {
  {
    auto l = [] { std::rcu_default_domain().lock(); };
    TEST_LIBCPP_ASSERT_FAILURE(
        std::jthread(l), "rcu_domain::unlock must be called before exiting a thread if rcu_domain::lock is called");
  }

  return 0;
}
