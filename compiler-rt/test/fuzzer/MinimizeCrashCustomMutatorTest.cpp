// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that -minimize_crash passes the size returned by the previous mutation
// to the next one when mutate_depth > 1.
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "FuzzerInterface.h"

static volatile int *Null = 0;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size > 0 && Data[0] == '!')
    *Null = 1;
  return 0;
}

// Maintains the invariant Data[0] == '0' + Size, so a stale Size passed by the
// minimizer is detected.
extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size,
                                          size_t MaxSize, unsigned int Seed) {
  if (Data[0] != '!' && Data[0] != '0' + Size)
    fprintf(stderr, "MUTATOR: size mismatch");
  Data[0] = '0' + MaxSize;
  return MaxSize;
}
