//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// REQUIRES: host-has-dbx

// RUN: %{cxx} %{flags} %s -o %t.exe %{compile_flags} -g %{link_flags}
// RUN: "%{dbx}" -c %S/is_debugger_present__dbx.cmd %t.exe \
// RUN:   | grep -qFf %S/is_debugger_present__dbx.grep

// <debugging>

// bool is_debugger_present() noexcept;

#include <cassert>
#include <concepts>
#include <debugging>

#include "test_macros.h"

#ifdef TEST_COMPILER_GCC
#  define OPT_NONE __attribute__((noinline))
#else
#  define OPT_NONE __attribute__((optnone))
#endif

// Prevents the compiler optimizing away the parameter in the caller function.
template <typename Type>
void MarkAsLive(Type&&) OPT_NONE;
template <typename Type>
void MarkAsLive(Type&&) {}

void StopForDebugger(void*) OPT_NONE;
void StopForDebugger(void*) {}

void test() {
  std::same_as<bool> decltype(auto) isDebuggerPresent = std::is_debugger_present();
  MarkAsLive(isDebuggerPresent);
  StopForDebugger(&isDebuggerPresent);

  static_assert(noexcept(std::is_debugger_present()));
}

int main(int, char**) {
  test();

  return 0;
}
