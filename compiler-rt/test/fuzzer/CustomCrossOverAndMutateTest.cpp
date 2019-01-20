// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that libFuzzer does not crash when LLVMFuzzerMutate called from
// LLVMFuzzerCustomCrossOver.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include <string>
#include <vector>

#include "FuzzerInterface.h"

static volatile int sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  std::string Str(reinterpret_cast<const char *>(Data), Size);
  if (Size && Data[0] == '0')
    sink++;
  return 0;
}

extern "C" size_t LLVMFuzzerCustomCrossOver(const uint8_t *Data1, size_t Size1,
                                            const uint8_t *Data2, size_t Size2,
                                            uint8_t *Out, size_t MaxOutSize,
                                            unsigned int Seed) {
  std::vector<uint8_t> Buffer(MaxOutSize * 10);
  LLVMFuzzerMutate(Buffer.data(), Buffer.size(), Buffer.size());
  size_t Size = std::min(Size1, MaxOutSize);
  memcpy(Out, Data1, Size);
  return Size;
}
