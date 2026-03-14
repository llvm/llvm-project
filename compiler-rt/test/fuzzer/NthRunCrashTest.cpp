// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Crash on the N-th execution.
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ostream>

static int Counter;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Counter++ == 1000) {
    std::cout << "BINGO; Found the target, exiting\n" << std::flush;
    exit(1);
  }
  return 0;
}

