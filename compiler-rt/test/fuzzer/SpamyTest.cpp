// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The test spams to stderr and stdout.
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  assert(Data);
  printf("PRINTF_STDOUT\n");
  fflush(stdout);
  fprintf(stderr, "PRINTF_STDERR\n");
  std::cout << "STREAM_COUT\n";
  std::cout.flush();
  std::cerr << "STREAM_CERR\n";
  return 0;
}

