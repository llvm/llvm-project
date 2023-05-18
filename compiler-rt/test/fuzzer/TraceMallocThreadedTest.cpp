// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Check that allocation tracing from different threads does not cause
// interleaving of stack traces.
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <thread>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  auto C = [&] {
    void * volatile a = malloc(5639);
    free((void *)a);
  };
  std::thread T[] = {std::thread(C), std::thread(C), std::thread(C),
                     std::thread(C), std::thread(C), std::thread(C)};
  for (auto &X : T)
    X.join();
  return 0;
}
