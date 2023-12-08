// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer. The fuzzer must find the string "Hi!".
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ostream>

static volatile bool SeedLargeBuffer;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  assert(Data);
  if (Size >= 4)
    SeedLargeBuffer = true;
  if (Size == 3 && SeedLargeBuffer && Data[3]) {
    std::cout << "Woops, reading Data[3] w/o crashing\n" << std::flush;
    exit(1);
  }
  return 0;
}

