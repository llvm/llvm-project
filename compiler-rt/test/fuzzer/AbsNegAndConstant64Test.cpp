// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// abs(x) < 0 and y == Const puzzle, 64-bit variant.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size < 16) return 0;
  int64_t x;
  uint64_t y;
  memcpy(&x, Data, sizeof(x));
  memcpy(&y, Data + sizeof(x), sizeof(y));
  volatile int64_t abs_x = llabs(x);
  if (abs_x < 0 && y == 0xbaddcafedeadbeefULL) {
    printf("BINGO; Found the target, exiting; x = 0x%lx y 0x%lx\n", x, y);
    fflush(stdout);
    exit(1);
  }
  return 0;
}

