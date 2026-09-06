//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: availability-debugging-missing

// <debugging>

// bool is_debugger_present() noexcept;

// Test without debugger attached

#include <cassert>
#include <debugging>

int main(int, char**) {
  assert(!std::is_debugger_present());
  static_assert(noexcept(std::is_debugger_present()));
  return 0;
}
