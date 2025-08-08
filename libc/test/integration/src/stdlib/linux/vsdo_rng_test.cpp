//===-- Test for vsdo_rng functionality ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/stdlib/linux/vsdo_rng.h"
#include "test/IntegrationTest/test.h"

using namespace LIBC_NAMESPACE;

void basic_test() {
  // Test basic functionality
  vsdo_rng::LocalState &local_state = vsdo_rng::local_state;

  // Try to get a guard
  if (auto guard = local_state.get()) {
    // Fill a small buffer with random data
    long long buffer[32] = {0};
    guard->fill(buffer, sizeof(buffer));

    // Basic sanity check - buffer should not have zero
    for (auto &i : buffer)
      if (i == 0)
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
