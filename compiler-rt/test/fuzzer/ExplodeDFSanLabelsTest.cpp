// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// When tracing data flow, explode the number of DFSan labels.
#include <cstddef>
#include <cstdint>

static volatile int sink;

__attribute__((noinline))
void f(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  if (a == b + 1 && c == d + 2)
    sink++;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  for (size_t a = 0; a < Size; a++)
    for (size_t b = 0; b < Size; b++)
      for (size_t c = 0; c < Size; c++)
        for (size_t d = 0; d < Size; d++)
          f(Data[a], Data[b], Data[c], Data[d]);
  return 0;
}
