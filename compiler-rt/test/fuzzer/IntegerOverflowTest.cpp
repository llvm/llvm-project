// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer. The fuzzer must find the string "Hi" and cause an
// integer overflow.
#include <cstddef>
#include <cstdint>

static int Val = 1 << 30;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size >= 2 && Data[0] == 'H' && Data[1] == 'i')
    Val += Val;
  return 0;
}

