// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Like SimpleTest, but simulates an "empty" module (i.e. one without any functions to instrument).
// This reproduces a previous bug (when libFuzzer is compiled with assertions enabled).

#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ostream>

extern "C" {
void __sanitizer_cov_8bit_counters_init(uint8_t *Start, uint8_t *Stop);
void __sanitizer_cov_pcs_init(const uintptr_t *pcs_beg,
                              const uintptr_t *pcs_end);
}

void dummy_func() {}

uint8_t empty_8bit_counters[0];
uintptr_t empty_pcs[0];

uint8_t fake_8bit_counters[1] = {0};
uintptr_t fake_pcs[2] = {reinterpret_cast<uintptr_t>(&dummy_func),
                         reinterpret_cast<uintptr_t>(&dummy_func)};

// Register two modules at program launch (same time they'd normally be registered).
// Triggering the bug requires loading an empty module, then a non-empty module after it.
bool dummy = []() {
  // First, simulate loading an empty module.
  __sanitizer_cov_8bit_counters_init(empty_8bit_counters, empty_8bit_counters);
  __sanitizer_cov_pcs_init(empty_pcs, empty_pcs);

  // Next, simulate loading a non-empty module.
  __sanitizer_cov_8bit_counters_init(fake_8bit_counters,
                                     fake_8bit_counters + 1);
  __sanitizer_cov_pcs_init(fake_pcs, fake_pcs + 2);

  return true;
}();

static volatile int Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  assert(Data);
  if (Size > 0 && Data[0] == 'H') {
    Sink = 1;
    if (Size > 1 && Data[1] == 'i') {
      Sink = 2;
      if (Size > 2 && Data[2] == '!') {
        std::cout << "BINGO; Found the target, exiting\n" << std::flush;
        exit(0);
      }
    }
  }
  return 0;
}
