// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test with a leak.
#include <cstddef>
#include <cstdint>

static volatile int *Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (!Size) return 0;
  Sink = new int;
  Sink = new int;
  while (Sink) *Sink = 0;  // Infinite loop.
  return 0;
}

