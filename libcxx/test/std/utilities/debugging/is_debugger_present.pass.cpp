//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// XFAIL: LIBCXX-AIX-FIXME, LIBCXX-PICOLIBC-FIXME

// <debugging>

// bool is_debugger_present() noexcept;

#include <cassert>
#include <concepts>
#include <debugging>

// Test without debugger.

void test() {
  static_assert(noexcept(std::is_debugger_present()));

  std::same_as<bool> decltype(auto) isDebuggerPresent = is_debugger_present();
  assert(isDebuggerPresent == false);
}

int main(int, char**) {
  test();

  return 0;
}
