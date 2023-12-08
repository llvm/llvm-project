// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Threaded test for a fuzzer. The fuzzer should not crash.
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size < 8) return 0;
  assert(Data);
  auto C = [&] {
    size_t Res = 0;
    for (size_t i = 0; i < Size / 2; i++)
      Res += memcmp(Data, Data + Size / 2, 4);
    return Res;
  };
  std::thread T[] = {std::thread(C), std::thread(C), std::thread(C),
                     std::thread(C), std::thread(C), std::thread(C)};
  for (auto &X : T)
    X.join();
  return 0;
}

