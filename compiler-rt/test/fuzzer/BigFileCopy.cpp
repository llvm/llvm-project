// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "FuzzerIO.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  const char *FileName = "big-file.txt";
  FILE *f = fopen(FileName, "w");

  // This is the biggest file possible unless CopyFileToErr() uses Puts()
  fprintf(f, "%2147483646s", "2Gb-2");

  // This makes the file too big if CopyFileToErr() uses fprintf("%s", <file>)
  fprintf(f, "THIS LINE RESPONSIBLE FOR EXCEEDING 2Gb FILE SIZE\n");
  fclose(f);

  // Should now because CopyFileToErr() now uses Puts()
  fuzzer::CopyFileToErr(FileName);

  // File is >2Gb so clean up
  remove(FileName);

  return 0;
}
