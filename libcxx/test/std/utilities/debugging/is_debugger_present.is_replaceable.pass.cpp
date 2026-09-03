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

// void is_debugger_present() noexcept;

// Test that we can replace the std::is_debugger_present() by defining our own.

#include <cassert>
#include <debugging>

#ifdef _WIN32
#  define DLLIMPORT __declspec(dllimport)
#else
#  define DLLIMPORT
#endif

static int canary = 0;

DLLIMPORT bool std::is_debugger_present() noexcept {
  canary = 1;
  return true;
}

int main(int, char**) {
  assert(std::is_debugger_present());
  assert(canary == 1);

  return 0;
}
