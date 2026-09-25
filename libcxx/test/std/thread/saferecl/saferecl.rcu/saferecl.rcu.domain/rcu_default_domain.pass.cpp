//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// REQUIRES: std-at-least-c++26

// rcu_domain& rcu_default_domain() noexcept;

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <latch>
#include <rcu>
#include <thread>
#include <utility>

#include "make_test_thread.h"
#include "test_macros.h"

static_assert(noexcept(std::rcu_default_domain()));

int main(int, char**) {
  {
    std::same_as<std::rcu_domain&> decltype(auto) dom1 = std::rcu_default_domain();
    std::same_as<std::rcu_domain&> decltype(auto) dom2 = std::rcu_default_domain();
    assert(&dom1 == &dom2);
    dom1.lock();
    dom1.unlock();
  }

  return 0;
}
