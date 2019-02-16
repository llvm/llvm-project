// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Test for implicit-integer-sign-change.
#include <assert.h>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

static volatile uint32_t Sink;
static volatile int32_t Storage = -1;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  assert(Data);
  if (Size > 0 && Data[0] == 'H') {
    Sink = 1;
    if (Size > 1 && Data[1] == 'i') {
    Sink = 2;
      if (Size > 2 && Data[2] == '!') {
        Sink = Storage;  // 'sign change'.
      }
    }
  }
  return 0;
}
