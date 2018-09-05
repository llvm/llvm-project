// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Tests that deadlocks do not occur when an OOM occurs during symbolization.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "Bingo.h"

volatile unsigned Sink = 0;

// Do not inline this function.  We want to trigger NEW_FUNC symbolization when
// libFuzzer finds this function.  We use a macro to make the name as long
// possible, hoping to increase the time spent in symbolization and increase the
// chances of triggering a deadlock.
__attribute__((noinline)) void BINGO() {
  // Busy work.  Inserts a delay here so the deadlock is more likely to trigger.
  for (unsigned i = 0; i < 330000000; i++) Sink += i;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  assert(Data);
  if (Size < 3) return 0;
  if (Data[0] == 'F' &&
      Data[1] == 'U' &&
      Data[2] == 'Z')
    BINGO();
  return 0;
}

