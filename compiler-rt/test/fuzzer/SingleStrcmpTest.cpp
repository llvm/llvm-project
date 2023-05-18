// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer. The fuzzer must find a particular string.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size >= 7) {
    char Copy[7];
    memcpy(Copy, Data, 6);
    Copy[6] = 0;
    if (!strcmp(Copy, "qwerty")) {
      fprintf(stderr, "BINGO\n");
      exit(1);
    }
  }
  return 0;
}
