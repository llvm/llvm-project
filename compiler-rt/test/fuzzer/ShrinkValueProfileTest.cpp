// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that we can find the minimal item in the corpus (4 bytes: "FUZZ").
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static volatile uint32_t Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size < sizeof(uint32_t)) return 0;
  uint32_t X, Y;
  size_t Offset = Size < 8 ? 0 : Size / 2;
  memcpy(&X, Data + Offset, sizeof(uint32_t));
  memcpy(&Y, "FUZZ", sizeof(uint32_t));
  Sink = X == Y;
  return 0;
}

