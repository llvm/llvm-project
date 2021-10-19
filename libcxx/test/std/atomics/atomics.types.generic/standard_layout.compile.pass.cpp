//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03

// <atomic>

#include <atomic>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "atomic_helpers.h"

template <class Tp>
struct CheckStandardLayout {
  void operator()() const {
    typedef std::atomic<Tp> Atomic;
    static_assert(std::is_standard_layout<Atomic>::value, "");
  }
};

int main(int, char**) {
  TestEachIntegralType<CheckStandardLayout>()();
  TestEachFloatingPointType<CheckStandardLayout>()();
  TestEachPointerType<CheckStandardLayout>()();

  return 0;
}
