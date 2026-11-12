//===-- Integration test for CndVar with C11 threads ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/time_macros.h"
#include "src/__support/CPP/expected.h"
#include "src/__support/threads/CndVar.h"
#include "src/__support/threads/mutex.h"
#include "src/__support/threads/mutex_common.h"
#include "src/__support/threads/sleep.h"
#include "src/__support/time/clock_gettime.h"
#include "src/threads/thrd_create.h"
#include "src/threads/thrd_join.h"
#include "test/IntegrationTest/test.h"

namespace {

constexpr int THREAD_COUNT = 4;
constexpr int NUM_ITERATIONS = 10;

struct QueueState {
  LIBC_NAMESPACE::CndVar cnd{false};
  LIBC_NAMESPACE::Mutex m{false, false, false, false};
  size_t consumed = 0;
  size_t produced = 0;
  LIBC_NAMESPACE::cpp::Atomic<size_t> exited_consumers = 0;
  bool use_broadcast;
  bool allow_timeout;
};

void stress_test(bool use_broadcast, bool allow_timeout) {
  for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
    QueueState state{};
    state.use_broadcast = use_broadcast;
    state.allow_timeout = allow_timeout;
    constexpr size_t PRODUCER = THREAD_COUNT / 2;
    constexpr size_t CONSUMER = THREAD_COUNT - PRODUCER;
    constexpr size_t ITEMS_PER_PRODUCER = 200;
    thrd_t producer_threads[PRODUCER];
    thrd_t consumer_threads[CONSUMER];
    using TimeoutOpt =
        LIBC_NAMESPACE::cpp::optional<LIBC_NAMESPACE::CndVar::Timeout>;
    for (size_t i = 0; i < CONSUMER; ++i)
      ASSERT_EQ(LIBC_NAMESPACE::thrd_create(
                    &consumer_threads[i],
                    [](void *arg) {
                      auto *state = static_cast<QueueState *>(arg);
                      state->m.lock();
                      while (state->consumed != PRODUCER * ITEMS_PER_PRODUCER) {
                        TimeoutOpt timeout = LIBC_NAMESPACE::cpp::nullopt;
                        if (state->allow_timeout) {
                          timespec now{};
                          LIBC_NAMESPACE::internal::clock_gettime(
                              CLOCK_MONOTONIC, &now);
                          size_t sleep_ns = 1000;
                          now.tv_nsec += sleep_ns;
                          if (now.tv_nsec >= 1'000'000'000) {
                            now.tv_sec++;
                            now.tv_nsec -= 1'000'000'000;
                          }
                          timeout = TimeoutOpt(
                              LIBC_NAMESPACE::CndVar::Timeout::from_timespec(
                                  now,
                                  /*realtime=*/false)
                                  .value());
                        }
                        ASSERT_NE(state->cnd.wait(&state->m, timeout),
                                  LIBC_NAMESPACE::CndVarResult::MutexError);
                        if (state->produced == 0)
                          continue;
                        state->produced--;
                        state->consumed++;
                      }
                      state->m.unlock();
                      state->exited_consumers.fetch_add(1);
                      return 0;
                    },
                    &state),
                int(thrd_success));
    for (size_t i = 0; i < PRODUCER; ++i)
      ASSERT_EQ(LIBC_NAMESPACE::thrd_create(
                    &producer_threads[i],
                    [](void *arg) {
                      auto *state = static_cast<QueueState *>(arg);
                      for (size_t j = 0; j < ITEMS_PER_PRODUCER; ++j) {
                        state->m.lock();
                        state->produced++;
                        if (state->use_broadcast)
                          state->cnd.broadcast();
                        else
                          state->cnd.notify_one();
                        state->m.unlock();
                      }
                      return 0;
                    },
                    &state),
                int(thrd_success));

    // join producers
    for (size_t i = 0; i < PRODUCER; ++i)
      ASSERT_EQ(LIBC_NAMESPACE::thrd_join(producer_threads[i], nullptr),
                int(thrd_success));
    // keep signalling until all consumers have consumed all items
    while (state.exited_consumers != CONSUMER) {
      if (state.use_broadcast)
        state.cnd.broadcast();
      else
        state.cnd.notify_one();
    }
    // join consumers
    for (size_t i = 0; i < CONSUMER; ++i)
      ASSERT_EQ(LIBC_NAMESPACE::thrd_join(consumer_threads[i], nullptr),
                int(thrd_success));
  }
}
} // namespace

TEST_MAIN() {
  stress_test(false, false);
  stress_test(true, false);
  stress_test(false, true);
  stress_test(true, true);
  return 0;
}
