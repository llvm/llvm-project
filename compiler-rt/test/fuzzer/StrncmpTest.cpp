// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer. The fuzzer must find a particular string.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static volatile int sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  // TODO: check other sizes.
  const char *S = (const char*)Data;
  if (Size >= 8 && strncmp(S, "123", 8))
    sink = 1;
  if (Size >= 8 && strncmp(S, "01234567", 8) == 0) {
    if (Size >= 12 && strncmp(S + 8, "ABCD", 4) == 0) {
      if (Size >= 14 && strncmp(S + 12, "XY", 2) == 0) {
        if (Size >= 17 && strncmp(S + 14, "KLM", 3) == 0) {
          fprintf(stderr, "BINGO\n");
          exit(1);
        }
      }
    }
  }
  return 0;
}
