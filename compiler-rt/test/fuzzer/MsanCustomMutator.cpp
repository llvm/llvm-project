// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Triggers the bug described here:
// https://github.com/google/oss-fuzz/issues/4605
//
// Tests that custom mutators do not cause MSan false positives.  We are careful
// to use every parameter to ensure none cause false positives.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" {

int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) { return 0; }

size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size, size_t MaxSize,
                               unsigned int Seed) {
  if (Seed == 7)
    return 0;
  if (MaxSize == 0)
    return 0;
  for (size_t I = 0; I < Size; ++I) {
    if (Data[I] == 42) {
      printf("BINGO\n");
    }
  }
  return Size;
}

size_t LLVMFuzzerCustomCrossOver(
    const uint8_t *Data1, size_t Size1, const uint8_t *Data2, size_t Size2,
    uint8_t *Out, size_t MaxOutSize, unsigned int Seed) {
  if (Seed == 7)
    return 0;
  size_t I = 0;
  for (; I < Size1 && I < Size2 && I < MaxOutSize; ++I) {
    Out[I] = std::min(Data1[I], Data2[I]);
  }
  return I;
}

} // extern "C"
