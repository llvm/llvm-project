// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer.
// Here the target has a shallow OOM bug and a deeper crash.
// Make sure we can find the crash while ignoring OOMs.
#include <cstddef>
#include <cstdint>

static volatile int *Sink;
static volatile int *Zero;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size >= 3 && Data[0] == 'O' && Data[1] == 'O' && Data[2] == 'M')
    Sink = new int[1 << 28]; // instant OOM with -rss_limit_mb=128.
  if (Size >= 4 && Data[0] == 'F' && Data[1] == 'U' && Data[2] == 'Z' &&
      Data[3] == 'Z')  // a bit deeper crash.
    *Zero = 42;
  return 0;
}

