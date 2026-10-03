//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <rcu>

// Test hardening assertions for std::valarray.

// REQUIRES: can-test-hardening-assertions-extensive

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
