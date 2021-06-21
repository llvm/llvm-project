//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: no-exceptions

// <atomic>

#include <atomic>
#include <cassert>

struct throwing {
  throwing() { throw 42; }
};

int main(int, char**) {
  try {
    [[maybe_unused]] std::atomic<throwing> a;
    assert(false);
  } catch (int x) {
    assert(x == 42);
  }

  return 0;
}
