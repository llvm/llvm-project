// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests whether scaling the Entropic scheduling weight based on input execution
// time is effective or not. Inputs of size 10 will take at least 100
// microseconds more than any input of size 1-9. The input of size 2 in the
// corpus should be favored by the exec-time-scaled Entropic scheduling policy
// than the input of size 10 in the corpus, eventually finding the crashing
// input {0xab, 0xcd} with less executions.
#include <chrono>
#include <cstdint>
#include <thread>

static volatile int Sink;
static volatile int *Nil = nullptr;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size > 10)
    return 0; // To make the test quicker.

  if (Size != 2) {
    // execute a lot slower than the crashing input below.
    size_t ExecTimeUSec = 100;
    std::this_thread::sleep_for(std::chrono::microseconds(ExecTimeUSec));
    if (Size > 0 && Data[0] == 0xaa && Size > 1 && Data[1] == 0xbb &&
        Size > 2 && Data[2] == 0xcc && Size > 3 && Data[3] == 0xdd &&
        Size > 4 && Data[4] == 0xee && Size > 5 && Data[5] == 0xff)
      Sink += 7;
  }

  if (Size == 2 && Data[0] == 0xab && Data[1] == 0xcd)
    *Nil = 42; // crash.

  return 0;
}
