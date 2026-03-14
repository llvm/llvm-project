// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test the fuzzer is able to 'cleanse' the reproducer
// by replacing all irrelevant bytes with garbage.
#include <cstddef>
#include <cstdint>
#include <cstdlib>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size >= 20 && Data[1] == '1' && Data[5] == '5' && Data[10] == 'A' &&
      Data[19] == 'Z')
    abort();
  return 0;
}

