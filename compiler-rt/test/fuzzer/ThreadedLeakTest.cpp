// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The fuzzer should find a leak in a non-main thread.
#include <cstddef>
#include <cstdint>
#include <thread>

static int * volatile Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size == 0) return 0;
  if (Data[0] != 'F') return 0;
  std::thread T([&] { Sink = new int; });
  T.join();
  return 0;
}

