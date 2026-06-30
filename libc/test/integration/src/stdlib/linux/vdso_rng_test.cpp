//===-- Test for vdso_rng functionality ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/stdlib/linux/vdso_rng.h"
#include "test/IntegrationTest/test.h"

using namespace LIBC_NAMESPACE;

void basic_test() {
  // Test basic functionality
  vdso_rng::LocalState &local_state = vdso_rng::local_state;

  // Try to get a guard
  if (auto guard = local_state.get()) {
    // Fill a small buffer with random data
    char buffer[256]{};
    guard->fill(buffer, sizeof(buffer));

    // Basic sanity check - count zero bytes.
    // With 256 bytes, getting more than ~10 zero bytes would be suspicious
    size_t zero_count = 0;
    for (auto &i : buffer)
      if (i == 0)
        zero_count++;

    // With uniform distribution, expect ~1 zero byte per 256 bytes
    // Having more than 16 zero bytes in 256 bytes is very unlikely
    if (zero_count > 16)
      __builtin_trap();
  }
  // If we can't get a guard, that's okay - the vDSO might not be available
  // or the system might not support getrandom
}

void multithread_test() {
  constexpr static size_t OUTER_REPEAT = 8;
  constexpr static size_t INNER_REPEAT = 32;
  constexpr static size_t NUM_THREADS = 16;
  pthread_t threads[NUM_THREADS];

  // Repeat outer loop so that
  for (size_t r = 0; r < OUTER_REPEAT; ++r) {
    for (pthread_t &thread : threads)
      LIBC_NAMESPACE::pthread_create(
          &thread, nullptr,
          [](void *) -> void * {
            for (size_t j = 0; j < INNER_REPEAT; ++j)
              basic_test();
            return nullptr;
          },
          nullptr);
    for (pthread_t thread : threads)
      LIBC_NAMESPACE::pthread_join(thread, nullptr);
  }
}

TEST_MAIN() {
  basic_test();
  multithread_test();
  return 0;
}
