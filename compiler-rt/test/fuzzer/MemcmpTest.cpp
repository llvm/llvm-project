// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer. The fuzzer must find a particular string.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef MEMCMP
# define MEMCMP memcmp
#endif

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  // TODO: check other sizes.
  if (Size >= 8 && MEMCMP(Data, "01234567", 8) == 0) {
    if (Size >= 12 && MEMCMP(Data + 8, "ABCD", 4) == 0) {
      if (Size >= 14 && MEMCMP(Data + 12, "XY", 2) == 0) {
        if (Size >= 17 && MEMCMP(Data + 14, "KLM", 3) == 0) {
          if (Size >= 27 && MEMCMP(Data + 17, "ABCDE-GHIJ", 10) == 0){
            fprintf(stderr, "BINGO %zd\n", Size);
            for (size_t i = 0; i < Size; i++) {
              uint8_t C = Data[i];
              if (C >= 32 && C < 127)
                fprintf(stderr, "%c", C);
            }
            fprintf(stderr, "\n");
            exit(1);
          }
        }
      }
    }
  }
  return 0;
}
