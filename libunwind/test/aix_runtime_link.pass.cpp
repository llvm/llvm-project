//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that libunwind loads successfully independently of libc++abi with
// runtime linking on AIX.

// REQUIRES: target={{.+}}-aix{{.*}}
// ADDITIONAL_COMPILE_FLAGS: -Wl,-brtl

#include <unwind.h>
extern "C" int printf(const char *, ...);
int main(void) {
  void *fp = (void *)&_Unwind_Backtrace;
  printf("%p\n", fp);
}
