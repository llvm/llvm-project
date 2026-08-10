// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that we can find the minimal item in the corpus (3 bytes: "FUZ").
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static volatile int Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size < 2) return 0;
  if (Data[0] == 'F' && Data[Size / 2] == 'U' && Data[Size - 1] == 'Z')
    Sink++;
  return 0;
}

