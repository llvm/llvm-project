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
void *fake_thread_entry_point(void *) { return nullptr; }

/*
  The creat function doesn't seem to work on an ubuntu Docker image when the
  path is in a shared volume of the host. For now, to keep testing convenient
  with a local Docker container, we just put it somewhere that's not in the
  shared volume (/tmp). This is volatile and will be cleaned up as soon as the
  container is stopped.
*/
constexpr const char *temporary_file_path() {
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

TEST(TestRadsanInterceptors, mallocDiesWhenRealtime) {
  auto func = []() { EXPECT_NE(nullptr, malloc(1)); };
  ExpectRealtimeDeath(func, "malloc");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, reallocDiesWhenRealtime) {
  auto *ptr_1 = malloc(1);
  auto func = [ptr_1]() { EXPECT_NE(nullptr, realloc(ptr_1, 8)); };
  ExpectRealtimeDeath(func, "realloc");
  ExpectNonRealtimeSurvival(func);
}

#if SANITIZER_APPLE
TEST(TestRadsanInterceptors, reallocfDiesWhenRealtime) {
  auto *ptr_1 = malloc(1);
  auto func = [ptr_1]() { EXPECT_NE(nullptr, reallocf(ptr_1, 8)); };
  ExpectRealtimeDeath(func, "reallocf");
  ExpectNonRealtimeSurvival(func);
}
#endif

TEST(TestRadsanInterceptors, vallocDiesWhenRealtime) {
  auto func = []() { EXPECT_NE(nullptr, valloc(4)); };
  ExpectRealtimeDeath(func, "valloc");
  ExpectNonRealtimeSurvival(func);
}

#if SANITIZER_INTERCEPT_ALIGNED_ALLOC
TEST(TestRadsanInterceptors, alignedAllocDiesWhenRealtime) {
  auto func = []() { EXPECT_NE(nullptr, aligned_alloc(16, 32)); };
  ExpectRealtimeDeath(func, "aligned_alloc");
  ExpectNonRealtimeSurvival(func);
}
#endif

// free_sized and free_aligned_sized (both C23) are not yet supported
TEST(TestRadsanInterceptors, freeDiesWhenRealtime) {
  auto *ptr_1 = malloc(1);
  auto *ptr_2 = malloc(1);
  ExpectRealtimeDeath([ptr_1]() { free(ptr_1); }, "free");
  ExpectNonRealtimeSurvival([ptr_2]() { free(ptr_2); });

  // Prevent malloc/free pair being optimised out
  ASSERT_NE(nullptr, ptr_1);
  ASSERT_NE(nullptr, ptr_2);
}

TEST(TestRadsanInterceptors, freeSurvivesWhenRealtimeIfArgumentIsNull) {
  realtimeInvoke([]() { free(NULL); });
  ExpectNonRealtimeSurvival([]() { free(NULL); });
}

TEST(TestRadsanInterceptors, posixMemalignDiesWhenRealtime) {
  auto func = []() {
    void *mem;
    posix_memalign(&mem, 4, 4);
  };
  ExpectRealtimeDeath(func, "posix_memalign");
  ExpectNonRealtimeSurvival(func);
}

#if SANITIZER_INTERCEPT_MEMALIGN
TEST(TestRadsanInterceptors, memalignDiesWhenRealtime) {
  auto func = []() { EXPECT_NE(memalign(2, 2048), nullptr); };
  ExpectRealtimeDeath(func, "memalign");
  ExpectNonRealtimeSurvival(func);
}
#endif

#if SANITIZER_INTERCEPT_PVALLOC
TEST(TestRadsanInterceptors, pvallocDiesWhenRealtime) {
  auto func = []() { EXPECT_NE(pvalloc(2048), nullptr); };
  ExpectRealtimeDeath(func, "pvalloc");
  ExpectNonRealtimeSurvival(func);
}
#endif

/*
    Sleeping
*/

TEST(TestRadsanInterceptors, sleepDiesWhenRealtime) {
  auto func = []() { sleep(0u); };
  ExpectRealtimeDeath(func, "sleep");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, usleepDiesWhenRealtime) {
  auto func = []() { usleep(1u); };
  ExpectRealtimeDeath(func, "usleep");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, nanosleepDiesWhenRealtime) {
  auto func = []() {
    auto t = timespec{};
    nanosleep(&t, &t);
  };
  ExpectRealtimeDeath(func, "nanosleep");
  ExpectNonRealtimeSurvival(func);
}

/*
    Filesystem
*/

TEST(TestRadsanInterceptors, openDiesWhenRealtime) {
  auto func = []() { open(temporary_file_path(), O_RDONLY); };
  ExpectRealtimeDeath(func, "open");
  ExpectNonRealtimeSurvival(func);
  std::remove(temporary_file_path());
}

TEST(TestRadsanInterceptors, openatDiesWhenRealtime) {
  auto func = []() { openat(0, temporary_file_path(), O_RDONLY); };
  ExpectRealtimeDeath(func, "openat");
  ExpectNonRealtimeSurvival(func);
  std::remove(temporary_file_path());
}

TEST(TestRadsanInterceptors, creatDiesWhenRealtime) {
  auto func = []() { creat(temporary_file_path(), S_IWOTH | S_IROTH); };
  ExpectRealtimeDeath(func, "creat");
  ExpectNonRealtimeSurvival(func);
  std::remove(temporary_file_path());
}

TEST(TestRadsanInterceptors, fcntlDiesWhenRealtime) {
  auto func = []() { fcntl(0, F_GETFL); };
  ExpectRealtimeDeath(func, "fcntl");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, closeDiesWhenRealtime) {
  auto func = []() { close(0); };
  ExpectRealtimeDeath(func, "close");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, fopenDiesWhenRealtime) {
  auto func = []() {
    auto fd = fopen(temporary_file_path(), "w");
    EXPECT_THAT(fd, Ne(nullptr));
  };
  ExpectRealtimeDeath(func, "fopen");
  ExpectNonRealtimeSurvival(func);
  std::remove(temporary_file_path());
}

TEST(TestRadsanInterceptors, freadDiesWhenRealtime) {
  auto fd = fopen(temporary_file_path(), "w");
  auto func = [fd]() {
    char c{};
    fread(&c, 1, 1, fd);
  };
  ExpectRealtimeDeath(func, "fread");
  ExpectNonRealtimeSurvival(func);
  if (fd != nullptr)
    fclose(fd);
  std::remove(temporary_file_path());
}

TEST(TestRadsanInterceptors, fwriteDiesWhenRealtime) {
  auto fd = fopen(temporary_file_path(), "w");
  ASSERT_NE(nullptr, fd);
  auto message = "Hello, world!";
  auto func = [&]() { fwrite(&message, 1, 4, fd); };
  ExpectRealtimeDeath(func, "fwrite");
  ExpectNonRealtimeSurvival(func);
  std::remove(temporary_file_path());
}

TEST(TestRadsanInterceptors, fcloseDiesWhenRealtime) {
  auto fd = fopen(temporary_file_path(), "w");
  EXPECT_THAT(fd, Ne(nullptr));
  auto func = [fd]() { fclose(fd); };
  ExpectRealtimeDeath(func, "fclose");
  ExpectNonRealtimeSurvival(func);
  std::remove(temporary_file_path());
}

TEST(TestRadsanInterceptors, putsDiesWhenRealtime) {
  auto func = []() { puts("Hello, world!\n"); };
  ExpectRealtimeDeath(func);
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, fputsDiesWhenRealtime) {
  auto fd = fopen(temporary_file_path(), "w");
  ASSERT_THAT(fd, Ne(nullptr)) << errno;
  auto func = [fd]() { fputs("Hello, world!\n", fd); };
  ExpectRealtimeDeath(func);
  ExpectNonRealtimeSurvival(func);
  if (fd != nullptr)
    fclose(fd);
  std::remove(temporary_file_path());
}

/*
    Concurrency
*/

TEST(TestRadsanInterceptors, pthreadCreateDiesWhenRealtime) {
  auto func = []() {
    auto thread = pthread_t{};
    auto const attr = pthread_attr_t{};
    struct thread_info *tinfo;
    pthread_create(&thread, &attr, &fake_thread_entry_point, tinfo);
  };
  ExpectRealtimeDeath(func, "pthread_create");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, pthreadMutexLockDiesWhenRealtime) {
  auto func = []() {
    auto mutex = pthread_mutex_t{};
    pthread_mutex_lock(&mutex);
  };

  ExpectRealtimeDeath(func, "pthread_mutex_lock");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, pthreadMutexUnlockDiesWhenRealtime) {
  auto func = []() {
    auto mutex = pthread_mutex_t{};
    pthread_mutex_unlock(&mutex);
  };

  ExpectRealtimeDeath(func, "pthread_mutex_unlock");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, pthreadMutexJoinDiesWhenRealtime) {
  auto func = []() {
    auto thread = pthread_t{};
    pthread_join(thread, nullptr);
  };

  ExpectRealtimeDeath(func, "pthread_join");
  ExpectNonRealtimeSurvival(func);
}

#if SANITIZER_APPLE

#pragma clang diagnostic push
// OSSpinLockLock is deprecated, but still in use in libc++
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
TEST(TestRadsanInterceptors, osSpinLockLockDiesWhenRealtime) {
  auto func = []() {
    auto spin_lock = OSSpinLock{};
    OSSpinLockLock(&spin_lock);
  };
  ExpectRealtimeDeath(func, "OSSpinLockLock");
  ExpectNonRealtimeSurvival(func);
}
#pragma clang diagnostic pop

TEST(TestRadsanInterceptors, osUnfairLockLockDiesWhenRealtime) {
  auto func = []() {
    auto unfair_lock = os_unfair_lock_s{};
    os_unfair_lock_lock(&unfair_lock);
  };
  ExpectRealtimeDeath(func, "os_unfair_lock_lock");
  ExpectNonRealtimeSurvival(func);
}
#endif

#if SANITIZER_LINUX
TEST(TestRadsanInterceptors, spinLockLockDiesWhenRealtime) {
  auto spinlock = pthread_spinlock_t{};
  pthread_spin_init(&spinlock, PTHREAD_PROCESS_SHARED);
  auto func = [&]() { pthread_spin_lock(&spinlock); };
  ExpectRealtimeDeath(func, "pthread_spin_lock");
  ExpectNonRealtimeSurvival(func);
}
#endif

TEST(TestRadsanInterceptors, pthreadCondSignalDiesWhenRealtime) {
  auto func = []() {
    auto cond = pthread_cond_t{};
    pthread_cond_signal(&cond);
  };
  ExpectRealtimeDeath(func, "pthread_cond_signal");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, pthreadCondBroadcastDiesWhenRealtime) {
  auto func = []() {
    auto cond = pthread_cond_t{};
    pthread_cond_broadcast(&cond);
  };
  ExpectRealtimeDeath(func, "pthread_cond_broadcast");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, pthreadCondWaitDiesWhenRealtime) {
  auto cond = pthread_cond_t{};
  auto mutex = pthread_mutex_t{};
  ASSERT_EQ(0, pthread_cond_init(&cond, nullptr));
  ASSERT_EQ(0, pthread_mutex_init(&mutex, nullptr));
  auto func = [&]() { pthread_cond_wait(&cond, &mutex); };
  ExpectRealtimeDeath(func, "pthread_cond_wait");
  // It's very difficult to test the success case here without doing some
  // sleeping, which is at the mercy of the scheduler. What's really important
  // here is the interception - so we're only testing that for now.
}

TEST(TestRadsanInterceptors, pthreadRwlockRdlockDiesWhenRealtime) {
  auto func = []() {
    auto rwlock = pthread_rwlock_t{};
    pthread_rwlock_rdlock(&rwlock);
  };
  ExpectRealtimeDeath(func, "pthread_rwlock_rdlock");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, pthreadRwlockUnlockDiesWhenRealtime) {
  auto func = []() {
    auto rwlock = pthread_rwlock_t{};
    pthread_rwlock_unlock(&rwlock);
  };
  ExpectRealtimeDeath(func, "pthread_rwlock_unlock");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, pthreadRwlockWrlockDiesWhenRealtime) {
  auto func = []() {
    auto rwlock = pthread_rwlock_t{};
    pthread_rwlock_wrlock(&rwlock);
  };
  ExpectRealtimeDeath(func, "pthread_rwlock_wrlock");
  ExpectNonRealtimeSurvival(func);
}

/*
    Sockets
*/
TEST(TestRadsanInterceptors, openingASocketDiesWhenRealtime) {
  auto func = []() { socket(PF_INET, SOCK_STREAM, 0); };
  ExpectRealtimeDeath(func, "socket");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, sendToASocketDiesWhenRealtime) {
  auto func = []() { send(0, nullptr, 0, 0); };
  ExpectRealtimeDeath(func, "send");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, sendmsgToASocketDiesWhenRealtime) {
  auto const msg = msghdr{};
  auto func = [&]() { sendmsg(0, &msg, 0); };
  ExpectRealtimeDeath(func, "sendmsg");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, sendtoToASocketDiesWhenRealtime) {
  auto const addr = sockaddr{};
  auto const len = socklen_t{};
  auto func = [&]() { sendto(0, nullptr, 0, 0, &addr, len); };
  ExpectRealtimeDeath(func, "sendto");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, recvFromASocketDiesWhenRealtime) {
  auto func = []() { recv(0, nullptr, 0, 0); };
  ExpectRealtimeDeath(func, "recv");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, recvfromOnASocketDiesWhenRealtime) {
  auto addr = sockaddr{};
  auto len = socklen_t{};
  auto func = [&]() { recvfrom(0, nullptr, 0, 0, &addr, &len); };
  ExpectRealtimeDeath(func, "recvfrom");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, recvmsgOnASocketDiesWhenRealtime) {
  auto msg = msghdr{};
  auto func = [&]() { recvmsg(0, &msg, 0); };
  ExpectRealtimeDeath(func, "recvmsg");
  ExpectNonRealtimeSurvival(func);
}

TEST(TestRadsanInterceptors, shutdownOnASocketDiesWhenRealtime) {
  auto func = [&]() { shutdown(0, 0); };
  ExpectRealtimeDeath(func, "shutdown");
  ExpectNonRealtimeSurvival(func);
}
