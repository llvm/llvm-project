//===-- Tests for pthread_rwlock ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/atomic.h"
#include "src/__support/threads/sleep.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_rwlock_destroy.h"
#include "src/pthread/pthread_rwlock_init.h"
#include "src/pthread/pthread_rwlock_rdlock.h"
#include "src/pthread/pthread_rwlock_timedrdlock.h"
#include "src/pthread/pthread_rwlock_timedwrlock.h"
#include "src/pthread/pthread_rwlock_tryrdlock.h"
#include "src/pthread/pthread_rwlock_trywrlock.h"
#include "src/pthread/pthread_rwlock_unlock.h"
#include "src/pthread/pthread_rwlock_wrlock.h"
#include "src/pthread/pthread_rwlockattr_destroy.h"
#include "src/pthread/pthread_rwlockattr_init.h"
#include "src/pthread/pthread_rwlockattr_setkind_np.h"
#include "src/pthread/pthread_rwlockattr_setpshared.h"
#include "src/stdlib/exit.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/sys/random/getrandom.h"
#include "src/sys/wait/waitpid.h"
#include "src/time/clock_gettime.h"
#include "src/unistd/fork.h"
#include "test/IntegrationTest/test.h"
#include <errno.h>
#include <pthread.h>
#include <time.h>

static void smoke_test() {
  pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&rwlock, nullptr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_rdlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_tryrdlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_trywrlock(&rwlock), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_wrlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_rdlock(&rwlock), EDEADLK);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_wrlock(&rwlock), EDEADLK);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_tryrdlock(&rwlock), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_trywrlock(&rwlock), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&rwlock), 0);
}

static void deadlock_detection_test() {
  pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&rwlock, nullptr), 0);
  // We only detect RAW, WAW deadlocks.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_wrlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_wrlock(&rwlock), EDEADLK);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&rwlock), 0);
}

static void try_lock_test() {
  pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&rwlock, nullptr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_wrlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_trywrlock(&rwlock), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_tryrdlock(&rwlock), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_tryrdlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_rdlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_tryrdlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_trywrlock(&rwlock), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&rwlock), 0);
}

static void destroy_before_unlock_test() {
  pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&rwlock, nullptr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_wrlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&rwlock), EBUSY);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&rwlock), 0);
}

static void nullptr_test() {
  timespec ts = {};
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_rdlock(nullptr), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_wrlock(nullptr), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_timedrdlock(nullptr, &ts), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_timedwrlock(nullptr, &ts), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_tryrdlock(nullptr), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_trywrlock(nullptr), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(nullptr), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(nullptr), EINVAL);
}

// If you are a user reading this code, please do not do something like this.
// We manually modify the internal state of the rwlock to test high reader
// counts.
static void high_reader_count_test() {
  pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;
  rwlock.__state = 0b01111111'11111111'11111111'11111100;
  //                 ^                                ^^
  //                 |                                ||
  //                 +-- writer bit                   ++-- pending bits
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_rdlock(&rwlock), EAGAIN);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_tryrdlock(&rwlock), EAGAIN);
  // allocate 4 reader slots.
  rwlock.__state -= 4 * 4;
  pthread_t threads[20];
  for (auto &i : threads) {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_create(
                  &i, nullptr,
                  [](void *arg) -> void * {
                    pthread_rwlock_t *rwlock =
                        reinterpret_cast<pthread_rwlock_t *>(arg);
                    ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_trywrlock(rwlock),
                              EBUSY);
                    while (LIBC_NAMESPACE::pthread_rwlock_rdlock(rwlock) ==
                           EAGAIN)
                      LIBC_NAMESPACE::sleep_briefly();
                    ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(rwlock), 0);
                    return nullptr;
                  },
                  &rwlock),
              0);
  }
  for (auto &i : threads) {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_join(i, nullptr), 0);
  }
}

static void unusual_timespec_test() {
  pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;
  timespec ts = {0, -1};
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_timedrdlock(&rwlock, &ts), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_timedwrlock(&rwlock, &ts), EINVAL);
  ts.tv_nsec = 1'000'000'000;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_timedrdlock(&rwlock, &ts), EINVAL);
  ts.tv_nsec += 1;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_timedwrlock(&rwlock, &ts), EINVAL);
  ts.tv_nsec = 0;
  ts.tv_sec = -1;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_timedrdlock(&rwlock, &ts),
            ETIMEDOUT);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_timedwrlock(&rwlock, &ts),
            ETIMEDOUT);
}

static void timedlock_with_deadlock_test() {
  pthread_rwlock_t rwlock = PTHREAD_RWLOCK_INITIALIZER;
  timespec ts{};
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_rdlock(&rwlock), 0);
  LIBC_NAMESPACE::clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec += 1;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_timedwrlock(&rwlock, &ts),
            ETIMEDOUT);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_timedrdlock(&rwlock, &ts), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  // notice that ts is already expired, but the following should still succeed.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_tryrdlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_trywrlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_rdlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_wrlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&rwlock), 0);
}

static void attributed_initialization_test() {
  pthread_rwlockattr_t attr{};
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_init(&attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_setkind_np(
                &attr, PTHREAD_RWLOCK_PREFER_READER_NP),
            0);
  {
    pthread_rwlock_t rwlock{};
    ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&rwlock, &attr), 0);
    ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&rwlock), 0);
  }
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_setkind_np(
                &attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP),
            0);
  {
    pthread_rwlock_t rwlock{};
    ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&rwlock, &attr), 0);
    ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&rwlock), 0);
  }
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_setkind_np(
                &attr, PTHREAD_RWLOCK_PREFER_WRITER_NP),
            0);
  {
    pthread_rwlock_t rwlock{};
    ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&rwlock, &attr), EINVAL);
  }
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_setkind_np(
                &attr, PTHREAD_RWLOCK_PREFER_READER_NP),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_setpshared(
                &attr, PTHREAD_PROCESS_PRIVATE),
            0);
  {
    pthread_rwlock_t rwlock{};
    ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&rwlock, &attr), 0);
    ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&rwlock), 0);
  }
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_setpshared(
                &attr, PTHREAD_PROCESS_SHARED),
            0);
  {
    pthread_rwlock_t rwlock{};
    ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&rwlock, &attr), 0);
    ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&rwlock), 0);
  }
  attr.pref = -1;
  {
    pthread_rwlock_t rwlock{};
    ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&rwlock, &attr), EINVAL);
  }
  attr.pref = PTHREAD_RWLOCK_PREFER_READER_NP;
  attr.pshared = -1;
  {
    pthread_rwlock_t rwlock{};
    ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&rwlock, &attr), EINVAL);
  }
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_destroy(&attr), 0);
}

struct SharedData {
  pthread_rwlock_t lock;
  int data;
  int reader_count;
  bool writer_flag;
  LIBC_NAMESPACE::cpp::Atomic<int> total_writer_count;
};

enum class Operation : int {
  READ,
  WRITE,
  TIMED_READ,
  TIMED_WRITE,
  TRY_READ,
  TRY_WRITE,
  COUNT
};

static void randomized_thread_operation(SharedData *data) {
  int buffer;
  // We cannot reason about thread order anyway, let's go wild and randomize it
  // directly using getrandom.
  LIBC_NAMESPACE::getrandom(&buffer, sizeof(buffer), 0);
  Operation op =
      static_cast<Operation>(buffer % static_cast<int>(Operation::COUNT));
  auto read_ops = [data]() {
    ASSERT_FALSE(data->writer_flag);
    ++data->reader_count;
    for (int i = 0; i < 10; ++i) {
      LIBC_NAMESPACE::sleep_briefly();
    }
    --data->reader_count;
  };
  auto write_ops = [data]() {
    ASSERT_FALSE(data->writer_flag);
    data->data += 1;
    data->writer_flag = true;
    for (int i = 0; i < 10; ++i) {
      LIBC_NAMESPACE::sleep_briefly();
    }
    data->writer_flag = false;
    data->total_writer_count.fetch_add(1);
  };
  auto get_ts = []() {
    timespec ts{};
    LIBC_NAMESPACE::clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_nsec += 5'000;
    if (ts.tv_nsec >= 1'000'000'000) {
      ts.tv_nsec -= 1'000'000'000;
      ts.tv_sec += 1;
    }
    return ts;
  };
  switch (op) {
  case Operation::READ: {
    LIBC_NAMESPACE::pthread_rwlock_rdlock(&data->lock);
    read_ops();
    LIBC_NAMESPACE::pthread_rwlock_unlock(&data->lock);
    break;
  }
  case Operation::WRITE: {
    LIBC_NAMESPACE::pthread_rwlock_wrlock(&data->lock);
    write_ops();
    LIBC_NAMESPACE::pthread_rwlock_unlock(&data->lock);
    break;
  }
  case Operation::TIMED_READ: {
    timespec ts = get_ts();
    if (LIBC_NAMESPACE::pthread_rwlock_timedrdlock(&data->lock, &ts) == 0) {
      read_ops();
      LIBC_NAMESPACE::pthread_rwlock_unlock(&data->lock);
    }
    break;
  }
  case Operation::TIMED_WRITE: {
    timespec ts = get_ts();
    if (LIBC_NAMESPACE::pthread_rwlock_timedwrlock(&data->lock, &ts) == 0) {
      write_ops();
      LIBC_NAMESPACE::pthread_rwlock_unlock(&data->lock);
    }
    break;
  }
  case Operation::TRY_READ: {
    if (LIBC_NAMESPACE::pthread_rwlock_tryrdlock(&data->lock) == 0) {
      read_ops();
      LIBC_NAMESPACE::pthread_rwlock_unlock(&data->lock);
    }
    break;
  }
  case Operation::TRY_WRITE: {
    if (LIBC_NAMESPACE::pthread_rwlock_trywrlock(&data->lock) == 0) {
      write_ops();
      LIBC_NAMESPACE::pthread_rwlock_unlock(&data->lock);
    }
    break;
  }
  case Operation::COUNT:
    __builtin_trap();
  }
}

static void
randomized_process_operation(SharedData &data,
                             LIBC_NAMESPACE::cpp::Atomic<int> &finish_count,
                             int expected_count) {
  pthread_t threads[32];
  for (auto &i : threads) {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_create(
                  &i, nullptr,
                  [](void *arg) -> void * {
                    randomized_thread_operation(
                        reinterpret_cast<SharedData *>(arg));
                    return nullptr;
                  },
                  &data),
              0);
  }
  for (auto &i : threads) {
    ASSERT_EQ(LIBC_NAMESPACE::pthread_join(i, nullptr), 0);
  }
  finish_count.fetch_add(1);
  while (finish_count.load() != expected_count) {
    LIBC_NAMESPACE::sleep_briefly();
  }
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&data.lock), 0);
  ASSERT_EQ(data.total_writer_count.load(), data.data);
  ASSERT_FALSE(data.writer_flag);
  ASSERT_EQ(data.reader_count, 0);
}

static void single_process_test(int preference) {
  SharedData data{};
  data.data = 0;
  data.reader_count = 0;
  data.writer_flag = false;
  data.total_writer_count.store(0);
  pthread_rwlockattr_t attr{};
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_init(&attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_setkind_np(&attr, preference),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&data.lock, nullptr), 0);
  LIBC_NAMESPACE::cpp::Atomic<int> finish_count{0};
  randomized_process_operation(data, finish_count, 1);
}

static void multiple_process_test(int preference) {
  struct PShared {
    SharedData data;
    LIBC_NAMESPACE::cpp::Atomic<int> finish_count;
  };
  PShared *shared_data = reinterpret_cast<PShared *>(
      LIBC_NAMESPACE::mmap(nullptr, sizeof(PShared), PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  pthread_rwlockattr_t attr{};
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_init(&attr), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_setkind_np(&attr, preference),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlockattr_setpshared(
                &attr, PTHREAD_PROCESS_SHARED),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_init(&shared_data->data.lock, &attr),
            0);
  int pid = LIBC_NAMESPACE::fork();
  randomized_process_operation(shared_data->data, shared_data->finish_count, 2);
  if (pid == 0) {
    LIBC_NAMESPACE::exit(0);
  } else {
    LIBC_NAMESPACE::waitpid(pid, nullptr, 0);
  }
  LIBC_NAMESPACE::munmap(shared_data, sizeof(PShared));
}

TEST_MAIN() {
  smoke_test();
  deadlock_detection_test();
  try_lock_test();
  destroy_before_unlock_test();
  nullptr_test();
  high_reader_count_test();
  unusual_timespec_test();
  timedlock_with_deadlock_test();
  attributed_initialization_test();
  single_process_test(PTHREAD_RWLOCK_PREFER_READER_NP);
  single_process_test(PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
  multiple_process_test(PTHREAD_RWLOCK_PREFER_READER_NP);
  multiple_process_test(PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
  return 0;
}
