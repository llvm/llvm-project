// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Contains dummy functions used to avoid dependency on AFL.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void __afl_manual_init() {}

extern "C" int __afl_persistent_loop(unsigned int N) {
  static int Count = N;
  fprintf(stderr, "__afl_persistent_loop calle, Count = %d\n", Count);
  if (Count--) return 1;
  return 0;
}

// This declaration exists to prevent the Darwin linker
// from complaining about this being a missing weak symbol.
extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
  fprintf(stderr, "LLVMFuzzerInitialize called\n");
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  fprintf(stderr, "LLVMFuzzerTestOneInput called; Size = %zd\n", Size);
  return 0;
}
