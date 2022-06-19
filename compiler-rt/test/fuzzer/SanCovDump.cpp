// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Link this tiny library to a binary compiled with
// -fsanitize-coverage=inline-8bit-counters.
// When passed SANCOV_OUT=OUTPUT_PATH, the process will
// dump the 8-bit coverage counters to the file, assuming
// the regular exit() is called.
//
// See OutOfProcessFuzzTarget.cpp for usage.
#include <stdio.h>
#include <stdlib.h>

static char *CovStart, *CovEnd;

static void DumpCoverage() {
  if (const char *DumpPath = getenv("SANCOV_OUT")) {
    fprintf(stderr, "SanCovDump: %p %p %s\n", CovStart, CovEnd, DumpPath);
    if (FILE *f = fopen(DumpPath, "w")) {
      fwrite(CovStart, 1, CovEnd - CovStart, f);
      fclose(f);
    }
  }
}

extern "C" void __sanitizer_cov_8bit_counters_init(char *Start, char *End) {
  CovStart = Start;
  CovEnd = End;
  atexit(DumpCoverage);
}
