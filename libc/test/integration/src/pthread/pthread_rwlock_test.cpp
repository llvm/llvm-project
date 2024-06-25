//===-- Tests for pthread_rwlock ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/new.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/threads/linux/raw_mutex.h"
#include "src/__support/threads/linux/rwlock.h"
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
#include "src/stdio/printf.h"
#include "src/stdlib/exit.h"
#include "src/stdlib/getenv.h"
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

namespace LIBC_NAMESPACE::rwlock {
class RwLockTester {
public:
  static constexpr int full_reader_state() {
    return (~0) & (~RwState::PENDING_MASK) & (~RwState::ACTIVE_WRITER_BIT);
  }
};
} // namespace LIBC_NAMESPACE::rwlock

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
  rwlock.__state = LIBC_NAMESPACE::rwlock::RwLockTester::full_reader_state();
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_rdlock(&rwlock), EAGAIN);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_tryrdlock(&rwlock), EAGAIN);
  // allocate 4 reader slots.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_unlock(&rwlock), 0);

  pthread_t threads[20];
  for (auto &i : threads)
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

  for (auto &i : threads)
    ASSERT_EQ(LIBC_NAMESPACE::pthread_join(i, nullptr), 0);
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
  ts.tv_nsec += 50'000;
  if (ts.tv_nsec >= 1'000'000'000) {
    ts.tv_nsec -= 1'000'000'000;
    ts.tv_sec += 1;
  }
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
  LIBC_NAMESPACE::cpp::Atomic<int> reader_count;
  bool writer_flag;
  LIBC_NAMESPACE::cpp::Atomic<int> total_writer_count;
};

enum class Operation : int {
  READ = 0,
  WRITE = 1,
  TIMED_READ = 2,
  TIMED_WRITE = 3,
  TRY_READ = 4,
  TRY_WRITE = 5,
  COUNT = 6
};

LIBC_NAMESPACE::RawMutex *io_mutex;
struct ThreadGuard {
  Operation record[64]{};
  size_t cursor = 0;
  void push(Operation op) { record[cursor++] = op; }
  ~ThreadGuard() {
    if (!LIBC_NAMESPACE::getenv("LIBC_PTHREAD_RWLOCK_TEST_VERBOSE"))
      return;
    pid_t pid = LIBC_NAMESPACE::syscall_impl(SYS_getpid);
    pid_t tid = LIBC_NAMESPACE::syscall_impl(SYS_gettid);
    io_mutex->lock(LIBC_NAMESPACE::cpp::nullopt, true);
    LIBC_NAMESPACE::printf("process %d thread %d: ", pid, tid);
    for (size_t i = 0; i < cursor; ++i)
      LIBC_NAMESPACE::printf("%d ", static_cast<int>(record[i]));
    LIBC_NAMESPACE::printf("\n");
    io_mutex->unlock(true);
  }
};

static void randomized_thread_operation(SharedData *data, ThreadGuard &guard) {
  int buffer;
  // We cannot reason about thread order anyway, let's go wild and randomize it
  // directly using getrandom.
  LIBC_NAMESPACE::getrandom(&buffer, sizeof(buffer), 0);
  constexpr int TOTAL = static_cast<int>(Operation::COUNT);
  Operation op = static_cast<Operation>(((buffer % TOTAL) + TOTAL) % TOTAL);
  guard.push(op);
  auto read_ops = [data]() {
    ASSERT_FALSE(data->writer_flag);
    data->reader_count.fetch_add(1, LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED);
    for (int i = 0; i < 10; ++i)
      LIBC_NAMESPACE::sleep_briefly();
    data->reader_count.fetch_sub(1, LIBC_NAMESPACE::cpp::MemoryOrder::RELAXED);
  };
  auto write_ops = [data]() {
    ASSERT_FALSE(data->writer_flag);
    data->data += 1;
    data->writer_flag = true;
    for (int i = 0; i < 10; ++i)
      LIBC_NAMESPACE::sleep_briefly();
    ASSERT_EQ(data->reader_count, 0);
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
  for (auto &i : threads)
    ASSERT_EQ(LIBC_NAMESPACE::pthread_create(
                  &i, nullptr,
                  [](void *arg) -> void * {
                    ThreadGuard guard{};
                    for (int i = 0; i < 64; ++i)
                      randomized_thread_operation(
                          reinterpret_cast<SharedData *>(arg), guard);
                    return nullptr;
                  },
                  &data),
              0);

  for (auto &i : threads)
    ASSERT_EQ(LIBC_NAMESPACE::pthread_join(i, nullptr), 0);

  finish_count.fetch_add(1);
  while (finish_count.load() != expected_count)
    LIBC_NAMESPACE::sleep_briefly();

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
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&data.lock), 0);
}

static void multiple_process_test(int preference) {
  struct PShared {
    SharedData data;
    LIBC_NAMESPACE::cpp::Atomic<int> finish_count;
  };
  PShared *shared_data = reinterpret_cast<PShared *>(
      LIBC_NAMESPACE::mmap(nullptr, sizeof(PShared), PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  shared_data->data.data = 0;
  shared_data->data.reader_count = 0;
  shared_data->data.writer_flag = false;
  shared_data->data.total_writer_count.store(0);
  shared_data->finish_count.store(0);
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
  if (pid == 0)
    LIBC_NAMESPACE::exit(0);
  else {
    int status;
    LIBC_NAMESPACE::waitpid(pid, &status, 0);
    ASSERT_EQ(status, 0);
  }
  ASSERT_EQ(LIBC_NAMESPACE::pthread_rwlock_destroy(&shared_data->data.lock), 0);
  LIBC_NAMESPACE::munmap(shared_data, sizeof(PShared));
}

TEST_MAIN() {
  io_mutex = new (LIBC_NAMESPACE::mmap(
      nullptr, sizeof(LIBC_NAMESPACE::RawMutex), PROT_READ | PROT_WRITE,
      MAP_ANONYMOUS | MAP_SHARED, -1, 0)) LIBC_NAMESPACE::RawMutex();
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
  io_mutex->~RawMutex();
  LIBC_NAMESPACE::munmap(io_mutex, sizeof(LIBC_NAMESPACE::RawMutex));
  return 0;
}
