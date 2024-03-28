//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// REQUIRES: host-has-gdb-with-python
// The Android libc++ tests are run on a non-Android host, connected to an
// Android device over adb. gdb needs special support to make this work (e.g.
// gdbclient.py, ndk-gdb.py, gdbserver), and the Android organization doesn't
// support gdb anymore, favoring lldb instead.
// UNSUPPORTED: android, linux && no-filesystem && no-localization
// XFAIL: LIBCXX-AIX-FIXME, LIBCXX-PICOLIBC-FIXME

// RUN: %{cxx} %{flags} %s -o %t.exe %{compile_flags} -g %{link_flags}
// RUN: "%{gdb}" %t.exe -ex "source %S/is_debugger_present_with_debugger_gdb.py" --silent

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
//   GDB command: `gdb is_debugger_present_with_debugger_gdb.sh -ex run -ex detach -ex quit --silent`

void test() {
  static_assert(noexcept(std::is_debugger_present()));

  std::same_as<bool> decltype(auto) isDebuggerPresent = std::is_debugger_present();
  MarkAsLive(isDebuggerPresent);
  StopForDebugger(&isDebuggerPresent);
}

int main(int, char**) {
  test();

  return 0;
}
