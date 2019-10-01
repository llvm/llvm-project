// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a cutom crossover.
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <random>
#include <string.h>
#include <functional>

static const char *Separator = "-########-";
static const char *Target = "A-########-B";

static volatile int sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  assert(Data);
  std::string Str(reinterpret_cast<const char *>(Data), Size);
  static const size_t TargetHash = std::hash<std::string>{}(std::string(Target));
  size_t StrHash = std::hash<std::string>{}(Str);

  // Ensure we have 'A' and 'B' in the corpus.
  if (Size == 1 && *Data == 'A')
    sink++;
  if (Size == 1 && *Data == 'B')
    sink--;

  if (TargetHash == StrHash) {
    std::cout << "BINGO; Found the target, exiting\n" << std::flush;
    exit(1);
  }
  return 0;
}

extern "C" size_t LLVMFuzzerCustomCrossOver(const uint8_t *Data1, size_t Size1,
                                            const uint8_t *Data2, size_t Size2,
                                            uint8_t *Out, size_t MaxOutSize,
                                            unsigned int Seed) {
  static size_t Printed;
  static size_t SeparatorLen = strlen(Separator);

  if (Printed++ < 32)
    std::cerr << "In LLVMFuzzerCustomCrossover " << Size1 << " " << Size2 << "\n";

  size_t Size = Size1 + Size2 + SeparatorLen;

  if (Size > MaxOutSize)
    return 0;

  memcpy(Out, Data1, Size1);
  memcpy(Out + Size1, Separator, SeparatorLen);
  memcpy(Out + Size1 + SeparatorLen, Data2, Size2);

  return Size;
}
