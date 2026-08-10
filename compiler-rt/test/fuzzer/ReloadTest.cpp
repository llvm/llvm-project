// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that fuzzer we can reload artifacts with any bytes inside.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <set>

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size,
                                          size_t MaxSize, unsigned int Seed) {
  std::srand(Seed);
  std::generate(Data, Data + MaxSize, std::rand);
  return MaxSize;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size > 5000 && std::set<uint8_t>(Data, Data + Size).size() > 255 &&
      (uint8_t)std::accumulate(Data, Data + Size, uint8_t(Size)) == 0)
    abort();
  return 0;
}
