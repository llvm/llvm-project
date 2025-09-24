//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// REQUIRES: host-has-lldb
// The Android libc++ tests are run on a non-Android host, connected to an
// Android device over adb.
// UNSUPPORTED: android
// XFAIL: LIBCXX-PICOLIBC-FIXME

// RUN: %{cxx} %{flags} %s -o %t.exe %{compile_flags} -g %{link_flags}
// RUN: "%{lldb}" %t.exe -o "command script import %S/is_debugger_present__lldb.py"

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

// Test with debugger attached:
//   LLDB command: `lldb "is_debugger_present_with_debugger__lldb.sh" -o run -o detach -o quit`

void test() {
  static_assert(noexcept(std::is_debugger_present()));

  std::same_as<bool> decltype(auto) isDebuggerPresent = std::is_debugger_present();
#if defined(TEST_HAS_NO_FILESYSTEM) || defined(_PICOLIB_)
  MarkAsLive(!isDebuggerPresent);
#else
  MarkAsLive(isDebuggerPresent);
#endif
  StopForDebugger(&isDebuggerPresent);
}

int main(int, char**) {
  test();

  return 0;
}
