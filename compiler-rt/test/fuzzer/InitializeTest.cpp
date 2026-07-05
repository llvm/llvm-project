// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Make sure LLVMFuzzerInitialize is called.
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *argv0 = NULL;

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
  assert(*argc > 0);
  argv0 = **argv;
  fprintf(stderr, "LLVMFuzzerInitialize: %s\n", argv0);
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  assert(argv0);
  if (argv0 && Size >= 4 && !memcmp(Data, "fuzz", 4)) {
    fprintf(stderr, "BINGO %s\n", argv0);
    exit(1);
  }
  return 0;
}
