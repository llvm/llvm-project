//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <debugging>

// bool is_debugger_present() noexcept;

#include <cassert>
#include <concepts>
#include <debugging>

// Test without a debugger.

void test() {
  std::same_as<bool> decltype(auto) isDebuggerPresent = std::is_debugger_present();
  assert(!isDebuggerPresent);

  static_assert(noexcept(std::is_debugger_present()));
}

int main(int, char**) {
  test();

  return 0;
}
