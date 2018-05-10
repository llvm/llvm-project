// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Find "FUZZME", the target has 3 different functions.
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

__attribute__((noinline))
static bool Func1(const uint8_t *Data, size_t Size) {
  // assumes Size >= 5, doesn't check it.
  return Data[4] == 'M';
}

__attribute__((noinline))
bool Func2(const uint8_t *Data, size_t Size) {
  return Size >= 6 && Data[5] == 'E';
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size >= 5
      && Data[0] == 'F'
      && Data[1] == 'U'
      && Data[2] == 'Z'
      && Data[3] == 'Z'
      && Func1(Data, Size)
      && Func2(Data, Size)) {
        fprintf(stderr, "BINGO\n");
        abort();
  }
  return 0;
}
