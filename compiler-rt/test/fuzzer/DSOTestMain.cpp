// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source code for a simple DSO.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
extern int DSO1(int a);
extern int DSO2(int a);
extern int DSOTestExtra(int a);

static volatile int *nil = 0;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  int x, y, z;
  if (Size < sizeof(int) * 3) {
    x = y = z = 0;
  } else {
    memcpy(&x, Data + 0 * sizeof(int), sizeof(int));
    memcpy(&y, Data + 1 * sizeof(int), sizeof(int));
    memcpy(&z, Data + 2 * sizeof(int), sizeof(int));
  }
  int sum = DSO1(x) + DSO2(y) + (z ? DSOTestExtra(z) : 0);
  if (sum == 3) {
    fprintf(stderr, "BINGO %d %d %d\n", x, y, z);
    *nil = 0;
  }
  return 0;
}
