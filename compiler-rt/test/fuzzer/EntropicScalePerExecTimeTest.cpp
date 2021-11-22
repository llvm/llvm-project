// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests whether scaling the Entropic scheduling weight based on input execution
// time is effective or not. Inputs of size less than 7 will take at least 100
// microseconds more than inputs of size greater than or equal to 7. Inputs of
// size greater than 7 in the corpus should be favored by the exec-time-scaled
// Entropic scheduling policy than the input of size less than 7 in the corpus,
// eventually finding the crashing input with less executions.
#include <chrono>
#include <cstdint>
#include <thread>

static volatile int Sink;
static volatile int *Nil = nullptr;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size > 10)
    return 0; // To make the test quicker.

  if (Size < 7) {
    // execute a lot slower than the crashing input below.
    size_t ExecTimeUSec = 100;
    std::this_thread::sleep_for(std::chrono::microseconds(ExecTimeUSec));
    Sink = 7;

    if (Size > 0 && Data[0] == 0xaa && Size > 1 && Data[1] == 0xbb &&
        Size > 2 && Data[2] == 0xcc && Size > 3 && Data[3] == 0xdd &&
        Size > 4 && Data[4] == 0xee && Size > 5 && Data[5] == 0xff)
      Sink += 7;
  }

  // Give unique coverage for each input of size (7, 8, 9, 10)
  if (Size == 7)
    Sink = -7;

  if (Size == 8)
    Sink = -8;

  if (Size == 9)
    Sink = -9;

  if (Size == 10)
    Sink = -10;

  if (Sink < 0 && Data[0] == 0xab && Data[1] == 0xcd)
    *Nil = 42; // crash.

  return 0;
}
