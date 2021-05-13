// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Stack overflow test for a fuzzer. The fuzzer must find the string "Hi" and
// cause a stack overflow.
#include <cstddef>
#include <cstdint>

volatile int x;
volatile int y = 1;

void infinite_recursion(char *p) {
  char *buf = nullptr;

  if (y)
    infinite_recursion(buf);

  x = 1;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size >= 2 && Data[0] == 'H' && Data[1] == 'i')
    infinite_recursion(nullptr);
  return 0;
}
