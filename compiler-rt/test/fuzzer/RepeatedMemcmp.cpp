// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  int Matches1 = 0;
  for (size_t i = 0; i + 2 < Size; i += 3)
    if (!memcmp(Data + i, "foo", 3))
      Matches1++;
  int Matches2 = 0;
  for (size_t i = 0; i + 2 < Size; i += 3)
    if (!memcmp(Data + i, "bar", 3))
      Matches2++;

  if (Matches1 > 10 && Matches2 > 10) {
    fprintf(stderr, "BINGO!\n");
    exit(1);
  }
  return 0;
}
