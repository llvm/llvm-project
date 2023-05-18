// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that libFuzzer itself does not read out of bounds.
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

static volatile int Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size < 5) return 0;
  const char *Ch = reinterpret_cast<const char *>(Data);
  if (Ch[Size - 3] == 'a')
    Sink = strncmp(Ch + Size - 3, "abcdefg", 6);
  return 0;
}

