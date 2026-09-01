//===-- race.h - Shared helpers for GPU ConcurrencySanitizer tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TSAN_GPU_TEST_RACE_H
#define TSAN_GPU_TEST_RACE_H

// Running total of races the device runtime has reported so far.
extern unsigned long long __tsan_num_data_races;

// True once the detector has fired at least once.
static inline int race_found(void) {
  return __atomic_load_n(&__tsan_num_data_races, __ATOMIC_RELAXED) != 0;
}

// Timeout value so the tests do not run forever.
#define RACE_MAX_ITERS (1 << 20)

// Sanitization reports are fundamentally probabalistic. We need to sample the
// racy region repeatedly until it fires.
#define RACE_UNTIL_FOUND(i)                                                    \
  for (int i = 0; i < RACE_MAX_ITERS && !race_found(); ++i)

#endif // TSAN_GPU_TEST_RACE_H
