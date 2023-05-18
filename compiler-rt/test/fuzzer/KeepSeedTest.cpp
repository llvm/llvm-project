// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test whether the fuzzer can find "SELECT FROM WHERE", given a seed input
// "SELECTxFROMxWHERE". Without -keep_seed=1, it takes longer time to trigger
// find the desired string, because the seed input is more likely to be reduced
// to a prefix of the given input first, losing useful fragments towards the end
// of the seed input.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static volatile int Sink = 0;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size > 17)
    return 0;

  if (Size >= 6 && Data[0] == 'S' && Data[1] == 'E' && Data[2] == 'L' &&
      Data[3] == 'E' && Data[4] == 'C' && Data[5] == 'T') {
    if (Size >= 7 && Data[6] == ' ') {
      if (Size >= 11 && Data[7] == 'F' && Data[8] == 'R' && Data[9] == 'O' &&
          Data[10] == 'M') {
        if (Size >= 12 && Data[11] == ' ') {
          if (Size >= 17 && Data[12] == 'W' && Data[13] == 'H' &&
              Data[14] == 'E' && Data[15] == 'R' && Data[16] == 'E') {
            fprintf(stderr, "BINGO; Found the target, exiting.\n");
            exit(1);
          }
        }
      }
    }
  }
  return 0;
}
