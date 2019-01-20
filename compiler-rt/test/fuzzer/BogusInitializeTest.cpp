// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Make sure LLVMFuzzerInitialize does not change argv[0].
#include <stddef.h>
#include <stdint.h>

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
  ***argv = 'X';
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  return 0;
}
