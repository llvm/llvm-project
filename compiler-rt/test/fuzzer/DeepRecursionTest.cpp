// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// Simple test for a fuzzer. The fuzzer must find the deep recursion.
// To generate a crashy input:
// for((i=0;i<110;i++)); do echo -n ABCDEFGHIJ  >> INPUT; done
#include <cstddef>
#include <cstdint>
#include <cstdlib>

static volatile int Sink;

void Recursive(const uint8_t *Data, size_t Size, int Depth) {
  if (Depth > 1000) abort();
  if (!Size) return;
  if (*Data == ('A' + Depth % 10))
    Recursive(Data + 1, Size - 1, Depth + 1);
  Sink++;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  Recursive(Data, Size, 0);
  return 0;
}

