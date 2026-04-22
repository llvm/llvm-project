//===-- Integration test for pthread condition variables ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/time_macros.h"
#include "src/__support/CPP/atomic.h"
#include "src/pthread/pthread_cond_broadcast.h"
#include "src/pthread/pthread_cond_clockwait.h"
#include "src/pthread/pthread_cond_destroy.h"
#include "src/pthread/pthread_cond_init.h"
#include "src/pthread/pthread_cond_signal.h"
#include "src/pthread/pthread_cond_timedwait.h"
#include "src/pthread/pthread_cond_wait.h"
#include "src/pthread/pthread_condattr_destroy.h"
#include "src/pthread/pthread_condattr_init.h"
#include "src/pthread/pthread_condattr_setpshared.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_mutex_destroy.h"
#include "src/pthread/pthread_mutex_init.h"
#include "src/pthread/pthread_mutex_lock.h"
#include "src/pthread/pthread_mutex_unlock.h"
#include "src/pthread/pthread_mutexattr_destroy.h"
#include "src/pthread/pthread_mutexattr_init.h"
#include "src/pthread/pthread_mutexattr_setpshared.h"
#include "src/time/clock_gettime.h"
#include "test/IntegrationTest/test.h"

#include <pthread.h>

namespace {

constexpr size_t PRODUCER_COUNT = 10;
constexpr size_t CONSUMER_COUNT = 10;
constexpr size_t ITEMS_PER_PRODUCER = 80;
constexpr int NUM_ITERATIONS = 10;

enum class TimeoutMode {
  Disabled,
  Default,
  Realtime,
  Monotonic,
};

template <bool UseBroadcast, bool IsShared, TimeoutMode Timeout>
struct TestConfig {
  LIBC_INLINE_VAR static constexpr bool USE_BROADCAST = UseBroadcast;
  LIBC_INLINE_VAR static constexpr bool IS_SHARED = IsShared;
  LIBC_INLINE_VAR static constexpr TimeoutMode TIMEOUT = Timeout;
};

struct QueueState {
  pthread_cond_t cond;
  pthread_mutex_t mutex;
  size_t consumed;
  size_t produced;
  LIBC_NAMESPACE::cpp::Atomic<size_t> exited_consumers;
};

static void add_ns(timespec &ts, long ns) {
  ts.tv_nsec += ns;
  if (ts.tv_nsec >= 1'000'000'000) {
    ++ts.tv_sec;
    ts.tv_nsec -= 1'000'000'000;
  }
}

template <typename Config> static timespec deadline() {
  static_assert(Config::TIMEOUT != TimeoutMode::Disabled);

  constexpr clockid_t CLOCK_ID = Config::TIMEOUT == TimeoutMode::Monotonic
                                     ? CLOCK_MONOTONIC
                                     : CLOCK_REALTIME;
  timespec ts{};
  ASSERT_EQ(LIBC_NAMESPACE::clock_gettime(CLOCK_ID, &ts), 0);
  add_ns(ts, 1000);
  return ts;
}

template <typename Config>
static int cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
  if constexpr (Config::TIMEOUT == TimeoutMode::Disabled) {
    return LIBC_NAMESPACE::pthread_cond_wait(cond, mutex);
  } else if constexpr (Config::TIMEOUT == TimeoutMode::Default) {
    timespec ts = deadline<Config>();
    return LIBC_NAMESPACE::pthread_cond_timedwait(cond, mutex, &ts);
  } else {
    constexpr clockid_t CLOCK_ID = Config::TIMEOUT == TimeoutMode::Monotonic
                                       ? CLOCK_MONOTONIC
                                       : CLOCK_REALTIME;
    timespec ts = deadline<Config>();
    return LIBC_NAMESPACE::pthread_cond_clockwait(cond, mutex, CLOCK_ID, &ts);
  }
}

template <typename Config> static void *consumer(void *arg) {
  auto *state = static_cast<QueueState *>(arg);
  constexpr size_t TOTAL_ITEMS = PRODUCER_COUNT * ITEMS_PER_PRODUCER;

  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&state->mutex), 0);
  while (state->consumed != TOTAL_ITEMS) {
    int wait_result = cond_wait<Config>(&state->cond, &state->mutex);
    if constexpr (Config::TIMEOUT == TimeoutMode::Disabled) {
      ASSERT_EQ(wait_result, 0);
    } else {
      ASSERT_TRUE(wait_result == 0 || wait_result == ETIMEDOUT);
    }

    if (state->produced == 0)
      continue;

    --state->produced;
    ++state->consumed;
  }
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&state->mutex), 0);
  state->exited_consumers.fetch_add(1);
  return nullptr;
}

template <typename Config> static void *producer(void *arg) {
  auto *state = static_cast<QueueState *>(arg);
  for (size_t i = 0; i < ITEMS_PER_PRODUCER; ++i) {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_lock(&state->mutex), 0);
    ++state->produced;
    if constexpr (Config::USE_BROADCAST) {
      ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_broadcast(&state->cond), 0);
    } else {
      ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_signal(&state->cond), 0);
    }
    ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_unlock(&state->mutex), 0);
  }
  return nullptr;
}

template <typename Config> static void init_state(QueueState &state) {
  state.consumed = 0;
  state.produced = 0;
  state.exited_consumers = 0;

  pthread_condattr_t cond_attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_condattr_init(&cond_attr), 0);
  pthread_mutexattr_t mutex_attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutexattr_init(&mutex_attr), 0);

  if constexpr (Config::IS_SHARED) {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_condattr_setpshared(
                  &cond_attr, PTHREAD_PROCESS_SHARED),
              0);
    ASSERT_EQ(LIBC_NAMESPACE::pthread_mutexattr_setpshared(
                  &mutex_attr, PTHREAD_PROCESS_SHARED),
              0);
  }

  ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_init(&state.cond, &cond_attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_init(&state.mutex, &mutex_attr), 0);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_condattr_destroy(&cond_attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutexattr_destroy(&mutex_attr), 0);
}

static void notify(QueueState &state, bool use_broadcast) {
  if (use_broadcast) {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_broadcast(&state.cond), 0);
  } else {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_signal(&state.cond), 0);
  }
}

template <typename Config> static void stress_test() {
  for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
    QueueState state{};
    init_state<Config>(state);

    pthread_t producer_threads[PRODUCER_COUNT];
    pthread_t consumer_threads[CONSUMER_COUNT];

    for (size_t i = 0; i < CONSUMER_COUNT; ++i)
      ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&consumer_threads[i], nullptr,
                                               consumer<Config>, &state),
                0);

    for (size_t i = 0; i < PRODUCER_COUNT; ++i)
      ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&producer_threads[i], nullptr,
                                               producer<Config>, &state),
                0);

    for (size_t i = 0; i < PRODUCER_COUNT; ++i)
      ASSERT_EQ(LIBC_NAMESPACE::pthread_join(producer_threads[i], nullptr), 0);

    while (state.exited_consumers != CONSUMER_COUNT)
      notify(state, Config::USE_BROADCAST);

    for (size_t i = 0; i < CONSUMER_COUNT; ++i)
      ASSERT_EQ(LIBC_NAMESPACE::pthread_join(consumer_threads[i], nullptr), 0);

    ASSERT_EQ(state.consumed, PRODUCER_COUNT * ITEMS_PER_PRODUCER);
    ASSERT_EQ(state.produced, size_t(0));
    ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_destroy(&state.cond), 0);
    ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_destroy(&state.mutex), 0);
  }
}

template <bool UseBroadcast, bool IsShared> static void run_timeout_modes() {
  stress_test<TestConfig<UseBroadcast, IsShared, TimeoutMode::Disabled>>();
  stress_test<TestConfig<UseBroadcast, IsShared, TimeoutMode::Default>>();
  stress_test<TestConfig<UseBroadcast, IsShared, TimeoutMode::Realtime>>();
  stress_test<TestConfig<UseBroadcast, IsShared, TimeoutMode::Monotonic>>();
}

template <bool UseBroadcast> static void run_shared_modes() {
  run_timeout_modes<UseBroadcast, false>();
  run_timeout_modes<UseBroadcast, true>();
}

void clockwait_returns_einval_for_invalid_clockid() {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_init(&mutex, nullptr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_init(&cond, nullptr), 0);

  timespec ts{};
  ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_clockwait(&cond, &mutex, -1, &ts),
            EINVAL);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_mutex_destroy(&mutex), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_destroy(&cond), 0);
}

void initializer_act_the_same_as_null_attr() {
  constexpr size_t EFFECTIVE_BYTES = sizeof(pthread_cond_t) - 2;
  union {
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    char cond_bytes[EFFECTIVE_BYTES];
  };
  union {
    pthread_cond_t cond_from_init;
    char cond_from_init_bytes[EFFECTIVE_BYTES];
  };
  ASSERT_EQ(LIBC_NAMESPACE::pthread_cond_init(&cond_from_init, nullptr), 0);
  // Read as bytes is a defined behavior for trivial types.
  for (size_t i = 0; i < EFFECTIVE_BYTES; ++i)
    ASSERT_EQ(cond_bytes[i], cond_from_init_bytes[i]);
}

} // namespace

TEST_MAIN() {
  run_shared_modes<false>();
  run_shared_modes<true>();
  initializer_act_the_same_as_null_attr();
  clockwait_returns_einval_for_invalid_clockid();
  return 0;
}
