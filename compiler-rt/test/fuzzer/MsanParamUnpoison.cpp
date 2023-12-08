// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Triggers the bug described here:
// https://github.com/google/oss-fuzz/issues/2369#issuecomment-490240627
//
// In a nutshell, MSan's parameter shadow does not get unpoisoned before calls
// to LLVMFuzzerTestOneInput.  This test case causes the parameter shadow to be
// poisoned by the call to foo(), which will trigger an MSan false positive on
// the Size == 0 check if the parameter shadow is still poisoned.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

volatile int zero = 0;
__attribute__((noinline)) int foo(int arg1, int arg2) { return zero; }

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size == 0)
    return 0;

  // Pass uninitialized values to foo().  Since foo doesn't do anything with
  // them, MSan should not report an error here.
  int a, b;
  return foo(a, b);
}
