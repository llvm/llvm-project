// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests how the fuzzer rejects inputs if the target returns -1.
#include <cstddef>
#include <cstdint>

static volatile int Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size != 3)
    return -1; // Reject anyting that's not 3 bytes long.
  // Reject 'rej'.
  if (Data[0] == 'r' && Data[1] == 'e' && Data[2] == 'j')
    return -1;
  // Accept 'acc'.
  if (Data[0] == 'a' && Data[1] == 'c' && Data[2] == 'c') {
    Sink = 1;
    return 0;
  }
  return 0;
}
