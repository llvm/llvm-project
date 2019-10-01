// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer.
// The fuzzer must find a string based on dictionary words:
//   "Elvis"
//   "Presley"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ostream>

static volatile int Zero = 0;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  const char *Expected = "ElvisPresley";
  if (Size < strlen(Expected)) return 0;
  size_t Match = 0;
  for (size_t i = 0; Expected[i]; i++)
    if (Expected[i] + Zero == Data[i])
      Match++;
  if (Match == strlen(Expected)) {
    std::cout << "BINGO; Found the target, exiting\n" << std::flush;
    exit(1);
  }
  return 0;
}

