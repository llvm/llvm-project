#include "src/__support/CPP/atomic.h"
#include "src/__support/libc_assert.h"
#include "src/__support/threads/linux/rwlock.h"
#include "src/errno/libc_errno.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "test/UnitTest/Test.h"
#include <linux/mman.h>
#include <llvm-libc-types/pid_t.h>
#include <pthread.h>
extern "C" pid_t fork(void);
extern "C" [[noreturn]] void exit(int);
namespace LIBC_NAMESPACE {
TEST(LlvmLibcRwLock, Smoke) {
  RwLock l{};
  l.read();
  l.read_unlock();
  l.write();
  l.write_unlock();
  l.read();
  l.read();
  l.read_unlock();
  l.read_unlock();
  l.write();
  l.write_unlock();
}

TEST(LlvmLibcRwLock, TrivialReadLock) {
  RwLock l{};
  pthread_t t[100];
  for (int i = 0; i < 100; i++) {
    pthread_create(
        &t[i], nullptr,
        [](void *arg) -> void * {
          RwLock *l = static_cast<RwLock *>(arg);
          if (!l->try_read())
            l->read();
          l->read_unlock();
          return nullptr;
        },
        &l);
  }
  for (int i = 0; i < 100; i++) {
    pthread_join(t[i], nullptr);
  }
}

TEST(LlvmLibcRwLock, TrivialWriteLock) {
  struct Data {
    RwLock lock{};
    cpp::Atomic<int> i = 0;
  } data;

  pthread_t t[100];
  for (int i = 0; i < 100; i++) {
    pthread_create(
        &t[i], nullptr,
        [](void *arg) -> void * {
          Data *data = static_cast<Data *>(arg);
          data->lock.write();
          int x = data->i.load(cpp::MemoryOrder::RELAXED);
          x++;
          data->i.store(x, cpp::MemoryOrder::RELAXED);
          data->lock.write_unlock();
          return nullptr;
        },
        &data);
  }
  for (int i = 0; i < 100; i++) {
    pthread_join(t[i], nullptr);
  }
  ASSERT_EQ(data.i.load(cpp::MemoryOrder::RELAXED), 100);
}

TEST(LlvmLibcRwLock, DeadWriteTimeoutRead) {
  RwLock l{};
  l.write();
  pthread_t t[100];
  for (int i = 0; i < 100; i++) {
    pthread_create(
        &t[i], nullptr,
        [](void *arg) -> void * {
          RwLock *l = static_cast<RwLock *>(arg);
          if (l->try_read())
            __builtin_trap();
          if (l->try_write())
            __builtin_trap();
          ::timespec ts{};
          ts.tv_sec = 1;
          if (l->read(ts))
            __builtin_trap();
          if (libc_errno != ETIMEDOUT)
            __builtin_trap();
          return nullptr;
        },
        &l);
  }
  for (int i = 0; i < 100; i++) {
    pthread_join(t[i], nullptr);
  }
}

TEST(LlvmLibcRwLock, DeadReadTimeoutWrite) {
  RwLock l{};
  l.read();
  pthread_t t[100];
  for (int i = 0; i < 100; i++) {
    pthread_create(
        &t[i], nullptr,
        [](void *arg) -> void * {
          RwLock *l = static_cast<RwLock *>(arg);
          // l->read_unlock();
          ::timespec ts{};
          ts.tv_sec = 1;
          if (l->try_write())
            __builtin_trap();
          if (l->write(ts))
            __builtin_trap();
          if (libc_errno != ETIMEDOUT)
            __builtin_trap();
          return nullptr;
        },
        &l);
  }
  for (int i = 0; i < 100; i++) {
    pthread_join(t[i], nullptr);
  }
}

TEST(LlvmLibcRwLock, Fork) {
  static constexpr struct Data {
    RwLock lock{true};
    cpp::Atomic<int> i = 0;
    cpp::Atomic<int> pending = 2;
  } DEFAULT;

  void *mmap_addr = mmap(nullptr, sizeof(Data), PROT_READ | PROT_WRITE,
                         MAP_ANONYMOUS | MAP_SHARED, -1, 0);
  ASSERT_NE(mmap_addr, MAP_FAILED);
  Data *data = static_cast<Data *>(mmap_addr);
  // placement new is not available as we cannot include <new>
  __builtin_memcpy(data, &DEFAULT, sizeof(Data));
  int pid = ::fork();

  pthread_t t[100];
  for (int i = 0; i < 100; i++) {
    pthread_create(
        &t[i], nullptr,
        [](void *arg) -> void * {
          Data *data = static_cast<Data *>(arg);
          data->lock.write();
          int x = data->i.load(cpp::MemoryOrder::RELAXED);
          x++;
          data->i.store(x, cpp::MemoryOrder::RELAXED);
          data->lock.write_unlock();
          return nullptr;
        },
        data);
  }
  for (int i = 0; i < 100; i++) {
    pthread_join(t[i], nullptr);
  }
  data->pending.fetch_sub(1);
  while (data->pending)
    /* not UB as operations are atomic */;
  ASSERT_EQ(data->i.load(cpp::MemoryOrder::RELAXED), 200);
  // early exit to avoid pollute the test output
  if (pid == 0)
    ::exit(0);
  ASSERT_EQ(munmap(mmap_addr, sizeof(Data)), 0);
}
} // namespace LIBC_NAMESPACE
