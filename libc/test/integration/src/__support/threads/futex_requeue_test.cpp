//===-- Integration test for futex requeue with real threads --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/threads/futex_utils.h"
#include "src/__support/threads/sleep.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "test/IntegrationTest/test.h"

#include <pthread.h>

namespace {

constexpr int NUM_ITERATIONS = 1000;

struct SharedState {
  LIBC_NAMESPACE::Futex lock1{1};
  LIBC_NAMESPACE::Futex lock2{1};
  LIBC_NAMESPACE::cpp::Atomic<int> started{0};
};

void wait_until_zero(LIBC_NAMESPACE::Futex &futex) {
  while (futex.load() != 0) {
    auto wait_result = futex.wait(1, LIBC_NAMESPACE::cpp::nullopt, false);
    ASSERT_TRUE(wait_result.has_value() || wait_result.error() == EAGAIN);
  }
}

void *worker(void *arg) {
  auto *state = reinterpret_cast<SharedState *>(arg);

  // This mimics a dual mutex handover process.
  state->started.store(1);
  wait_until_zero(state->lock1);
  wait_until_zero(state->lock2);

  return nullptr;
}

void futex_requeue_test() {
  for (int i = 0; i < NUM_ITERATIONS; ++i) {
    SharedState state;
    pthread_t thread;

    ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&thread, nullptr, worker, &state),
              0);

    while (state.started.load() == 0)
      LIBC_NAMESPACE::sleep_briefly();

    state.lock1.store(0);
    auto requeue_result = state.lock1.requeue_to(
        state.lock2, LIBC_NAMESPACE::FutexWordType(0), 0, 1, false);
    if (!requeue_result.has_value()) {
      ASSERT_EQ(requeue_result.error(), ENOSYS);
      auto wake_first = state.lock1.notify_one(false);
      ASSERT_TRUE(wake_first.has_value());
    }

    state.lock2.store(0);
    auto wake_second = state.lock2.notify_one(false);
    ASSERT_TRUE(wake_second.has_value());

    ASSERT_EQ(LIBC_NAMESPACE::pthread_join(thread, nullptr), 0);
  }
}

} // namespace

TEST_MAIN() {
  futex_requeue_test();
  return 0;
}
