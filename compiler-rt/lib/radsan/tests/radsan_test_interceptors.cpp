//===--- radsan_test_interceptors.cpp - Realtime Sanitizer --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include <sanitizer_common/sanitizer_platform.h>
#include <sanitizer_common/sanitizer_platform_interceptors.h>

#include "radsan_test_utilities.h"

#if SANITIZER_APPLE
#include <libkern/OSAtomic.h>
#include <os/lock.h>
#endif

#if SANITIZER_INTERCEPT_MEMALIGN || SANITIZER_INTERCEPT_PVALLOC
#include <malloc.h>
#endif

#include <atomic>
#include <chrono>
#include <thread>

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/socket.h>

using namespace testing;
using namespace radsan_testing;
using namespace std::chrono_literals;

namespace {
void *FakeThreadEntryPoint(void *) { return nullptr; }

/*
  The creat function doesn't seem to work on an ubuntu Docker image when the
  path is in a shared volume of the host. For now, to keep testing convenient
  with a local Docker container, we just put it somewhere that's not in the
  shared volume (/tmp). This is volatile and will be cleaned up as soon as the
  container is stopped.
*/
constexpr const char *TemporaryFilePath() {
#if SANITIZER_LINUX
  return "/tmp/radsan_temporary_test_file.txt";
#elif SANITIZER_APPLE
  return "./radsan_temporary_test_file.txt";
#endif
}
} // namespace

/*
    Allocation and deallocation
*/

TEST(TestRadsanInterceptors, MallocDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(nullptr, malloc(1)); };
  ExpectRealtimeDeath(Func, "malloc");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, ReallocDiesWhenRealtime) {
  void *ptr_1 = malloc(1);
  auto Func = [ptr_1]() { EXPECT_NE(nullptr, realloc(ptr_1, 8)); };
  ExpectRealtimeDeath(Func, "realloc");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_APPLE
TEST(TestRadsanInterceptors, ReallocfDiesWhenRealtime) {
  void *ptr_1 = malloc(1);
  auto Func = [ptr_1]() { EXPECT_NE(nullptr, reallocf(ptr_1, 8)); };
  ExpectRealtimeDeath(Func, "reallocf");
  ExpectNonRealtimeSurvival(Func);
}
#endif

TEST(TestRadsanInterceptors, VallocDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(nullptr, valloc(4)); };
  ExpectRealtimeDeath(Func, "valloc");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_INTERCEPT_ALIGNED_ALLOC
TEST(TestRadsanInterceptors, AlignedAllocDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(nullptr, aligned_alloc(16, 32)); };
  ExpectRealtimeDeath(Func, "aligned_alloc");
  ExpectNonRealtimeSurvival(Func);
}
#endif

// free_sized and free_aligned_sized (both C23) are not yet supported
TEST(TestRadsanInterceptors, FreeDiesWhenRealtime) {
  void *ptr_1 = malloc(1);
  void *ptr_2 = malloc(1);
  ExpectRealtimeDeath([ptr_1]() { free(ptr_1); }, "free");
  ExpectNonRealtimeSurvival([ptr_2]() { free(ptr_2); });

  // Prevent malloc/free pair being optimised out
  ASSERT_NE(nullptr, ptr_1);
  ASSERT_NE(nullptr, ptr_2);
}

TEST(TestRadsanInterceptors, FreeSurvivesWhenRealtimeIfArgumentIsNull) {
  RealtimeInvoke([]() { free(NULL); });
  ExpectNonRealtimeSurvival([]() { free(NULL); });
}

TEST(TestRadsanInterceptors, PosixMemalignDiesWhenRealtime) {
  auto Func = []() {
    void *Mem;
    posix_memalign(&Mem, 4, 4);
  };
  ExpectRealtimeDeath(Func, "posix_memalign");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_INTERCEPT_MEMALIGN
TEST(TestRadsanInterceptors, MemalignDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(memalign(2, 2048), nullptr); };
  ExpectRealtimeDeath(Func, "memalign");
  ExpectNonRealtimeSurvival(Func);
}
#endif

#if SANITIZER_INTERCEPT_PVALLOC
TEST(TestRadsanInterceptors, PvallocDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(pvalloc(2048), nullptr); };
  ExpectRealtimeDeath(Func, "pvalloc");
  ExpectNonRealtimeSurvival(Func);
}
#endif

/*
    Sleeping
*/

TEST(TestRadsanInterceptors, SleepDiesWhenRealtime) {
  auto Func = []() { sleep(0u); };
  ExpectRealtimeDeath(Func, "sleep");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, UsleepDiesWhenRealtime) {
  auto Func = []() { usleep(1u); };
  ExpectRealtimeDeath(Func, "usleep");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, NanosleepDiesWhenRealtime) {
  auto Func = []() {
    timespec T{};
    nanosleep(&T, &T);
  };
  ExpectRealtimeDeath(Func, "nanosleep");
  ExpectNonRealtimeSurvival(Func);
}

/*
    Filesystem
*/

TEST(TestRadsanInterceptors, OpenDiesWhenRealtime) {
  auto Func = []() { open(TemporaryFilePath(), O_RDONLY); };
  ExpectRealtimeDeath(Func, "open");
  ExpectNonRealtimeSurvival(Func);
  std::remove(TemporaryFilePath());
}

TEST(TestRadsanInterceptors, OpenatDiesWhenRealtime) {
  auto Func = []() { openat(0, TemporaryFilePath(), O_RDONLY); };
  ExpectRealtimeDeath(Func, "openat");
  ExpectNonRealtimeSurvival(Func);
  std::remove(TemporaryFilePath());
}

TEST(TestRadsanInterceptors, CreatDiesWhenRealtime) {
  auto Func = []() { creat(TemporaryFilePath(), S_IWOTH | S_IROTH); };
  ExpectRealtimeDeath(Func, "creat");
  ExpectNonRealtimeSurvival(Func);
  std::remove(TemporaryFilePath());
}

TEST(TestRadsanInterceptors, FcntlDiesWhenRealtime) {
  auto Func = []() { fcntl(0, F_GETFL); };
  ExpectRealtimeDeath(Func, "fcntl");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, CloseDiesWhenRealtime) {
  auto Func = []() { close(0); };
  ExpectRealtimeDeath(Func, "close");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, FopenDiesWhenRealtime) {
  auto Func = []() {
    FILE* Fd = fopen(TemporaryFilePath(), "w");
    EXPECT_THAT(Fd, Ne(nullptr));
  };
  ExpectRealtimeDeath(Func, "fopen");
  ExpectNonRealtimeSurvival(Func);
  std::remove(TemporaryFilePath());
}

TEST(TestRadsanInterceptors, FreadDiesWhenRealtime) {
  FILE* Fd = fopen(TemporaryFilePath(), "w");
  auto Func = [Fd]() {
    char c{};
    fread(&c, 1, 1, Fd);
  };
  ExpectRealtimeDeath(Func, "fread");
  ExpectNonRealtimeSurvival(Func);
  if (Fd != nullptr)
    fclose(Fd);
  std::remove(TemporaryFilePath());
}

TEST(TestRadsanInterceptors, FwriteDiesWhenRealtime) {
  FILE* Fd = fopen(TemporaryFilePath(), "w");
  ASSERT_NE(nullptr, Fd);
  const char* Message = "Hello, world!";
  auto Func = [&]() { fwrite(&Message, 1, 4, Fd); };
  ExpectRealtimeDeath(Func, "fwrite");
  ExpectNonRealtimeSurvival(Func);
  std::remove(TemporaryFilePath());
}

TEST(TestRadsanInterceptors, FcloseDiesWhenRealtime) {
  FILE* Fd = fopen(TemporaryFilePath(), "w");
  EXPECT_THAT(Fd, Ne(nullptr));
  auto Func = [Fd]() { fclose(Fd); };
  ExpectRealtimeDeath(Func, "fclose");
  ExpectNonRealtimeSurvival(Func);
  std::remove(TemporaryFilePath());
}

TEST(TestRadsanInterceptors, PutsDiesWhenRealtime) {
  auto Func = []() { puts("Hello, world!\n"); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, FputsDiesWhenRealtime) {
  FILE* Fd = fopen(TemporaryFilePath(), "w");
  ASSERT_THAT(Fd, Ne(nullptr)) << errno;
  auto Func = [Fd]() { fputs("Hello, world!\n", Fd); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
  if (Fd != nullptr)
    fclose(Fd);
  std::remove(TemporaryFilePath());
}

/*
    Concurrency
*/

TEST(TestRadsanInterceptors, PthreadCreateDiesWhenRealtime) {
  auto Func = []() {
    pthread_t Thread{};
    const pthread_attr_t Attr{};
    struct thread_info *ThreadInfo;
    pthread_create(&Thread, &Attr, &FakeThreadEntryPoint, ThreadInfo);
  };
  ExpectRealtimeDeath(Func, "pthread_create");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, PthreadMutexLockDiesWhenRealtime) {
  auto Func = []() {
    pthread_mutex_t Mutex{};
    pthread_mutex_lock(&Mutex);
  };

  ExpectRealtimeDeath(Func, "pthread_mutex_lock");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, PthreadMutexUnlockDiesWhenRealtime) {
  auto Func = []() {
    pthread_mutex_t Mutex{};
    pthread_mutex_unlock(&Mutex);
  };

  ExpectRealtimeDeath(Func, "pthread_mutex_unlock");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, PthreadMutexJoinDiesWhenRealtime) {
  auto Func = []() {
    pthread_t Thread{};
    pthread_join(Thread, nullptr);
  };

  ExpectRealtimeDeath(Func, "pthread_join");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_APPLE

#pragma clang diagnostic push
// OSSpinLockLock is deprecated, but still in use in libc++
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
TEST(TestRadsanInterceptors, OsSpinLockLockDiesWhenRealtime) {
  auto Func = []() {
    OSSpinLock SpinLock{};
    OSSpinLockLock(&SpinLock);
  };
  ExpectRealtimeDeath(Func, "OSSpinLockLock");
  ExpectNonRealtimeSurvival(Func);
}
#pragma clang diagnostic pop

TEST(TestRadsanInterceptors, OsUnfairLockLockDiesWhenRealtime) {
  auto Func = []() {
    os_unfair_lock_s UnfairLock{};
    os_unfair_lock_lock(&UnfairLock);
  };
  ExpectRealtimeDeath(Func, "os_unfair_lock_lock");
  ExpectNonRealtimeSurvival(Func);
}
#endif

#if SANITIZER_LINUX
TEST(TestRadsanInterceptors, SpinLockLockDiesWhenRealtime) {
  pthread_spinlock_t SpinLock;
  pthread_spin_init(&SpinLock, PTHREAD_PROCESS_SHARED);
  auto Func = [&]() { pthread_spin_lock(&SpinLock); };
  ExpectRealtimeDeath(Func, "pthread_spin_lock");
  ExpectNonRealtimeSurvival(Func);
}
#endif

TEST(TestRadsanInterceptors, PthreadCondSignalDiesWhenRealtime) {
  auto Func = []() {
    pthread_cond_t Cond{};
    pthread_cond_signal(&Cond);
  };
  ExpectRealtimeDeath(Func, "pthread_cond_signal");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, PthreadCondBroadcastDiesWhenRealtime) {
  auto Func = []() {
    pthread_cond_t Cond;
    pthread_cond_broadcast(&Cond);
  };
  ExpectRealtimeDeath(Func, "pthread_cond_broadcast");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, PthreadCondWaitDiesWhenRealtime) {
  pthread_cond_t Cond;
  pthread_mutex_t Mutex;
  ASSERT_EQ(0, pthread_cond_init(&Cond, nullptr));
  ASSERT_EQ(0, pthread_mutex_init(&Mutex, nullptr));
  auto Func = [&]() { pthread_cond_wait(&Cond, &Mutex); };
  ExpectRealtimeDeath(Func, "pthread_cond_wait");
  // It's very difficult to test the success case here without doing some
  // sleeping, which is at the mercy of the scheduler. What's really important
  // here is the interception - so we're only testing that for now.
}

TEST(TestRadsanInterceptors, PthreadRwlockRdlockDiesWhenRealtime) {
  auto Func = []() {
    pthread_rwlock_t RwLock;
    pthread_rwlock_rdlock(&RwLock);
  };
  ExpectRealtimeDeath(Func, "pthread_rwlock_rdlock");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, PthreadRwlockUnlockDiesWhenRealtime) {
  auto Func = []() {
    pthread_rwlock_t RwLock;
    pthread_rwlock_unlock(&RwLock);
  };
  ExpectRealtimeDeath(Func, "pthread_rwlock_unlock");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, PthreadRwlockWrlockDiesWhenRealtime) {
  auto Func = []() {
    pthread_rwlock_t RwLock;
    pthread_rwlock_wrlock(&RwLock);
  };
  ExpectRealtimeDeath(Func, "pthread_rwlock_wrlock");
  ExpectNonRealtimeSurvival(Func);
}

/*
    Sockets
*/
TEST(TestRadsanInterceptors, OpeningASocketDiesWhenRealtime) {
  auto Func = []() { socket(PF_INET, SOCK_STREAM, 0); };
  ExpectRealtimeDeath(Func, "socket");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, SendToASocketDiesWhenRealtime) {
  auto Func = []() { send(0, nullptr, 0, 0); };
  ExpectRealtimeDeath(Func, "send");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, SendmsgToASocketDiesWhenRealtime) {
  msghdr Msg{};
  auto Func = [&]() { sendmsg(0, &Msg, 0); };
  ExpectRealtimeDeath(Func, "sendmsg");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, SendtoToASocketDiesWhenRealtime) {
  sockaddr Addr{};
  socklen_t Len{};
  auto Func = [&]() { sendto(0, nullptr, 0, 0, &Addr, Len); };
  ExpectRealtimeDeath(Func, "sendto");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, RecvFromASocketDiesWhenRealtime) {
  auto Func = []() { recv(0, nullptr, 0, 0); };
  ExpectRealtimeDeath(Func, "recv");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, RecvfromOnASocketDiesWhenRealtime) {
  sockaddr Addr{};
  socklen_t Len{};
  auto Func = [&]() { recvfrom(0, nullptr, 0, 0, &Addr, &Len); };
  ExpectRealtimeDeath(Func, "recvfrom");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, RecvmsgOnASocketDiesWhenRealtime) {
  msghdr Msg{};
  auto Func = [&]() { recvmsg(0, &Msg, 0); };
  ExpectRealtimeDeath(Func, "recvmsg");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRadsanInterceptors, ShutdownOnASocketDiesWhenRealtime) {
  auto Func = [&]() { shutdown(0, 0); };
  ExpectRealtimeDeath(Func, "shutdown");
  ExpectNonRealtimeSurvival(Func);
}
