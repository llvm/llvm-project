// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests OOM handling.
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

static volatile char *SinkPtr;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size > 0 && Data[0] == 'H') {
    if (Size > 1 && Data[1] == 'i') {
      if (Size > 2 && Data[2] == '!') {
        while (true) {
          size_t kSize = 1 << 28;
          char *p = new char[kSize];
          memset(p, 0, kSize);
          SinkPtr = p;
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
      }
    }
  }
  return 0;
}

