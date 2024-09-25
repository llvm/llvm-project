//===--- rtsan_test_interceptors.cpp - Realtime Sanitizer -------*- C++ -*-===//
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

#include "rtsan_test_utilities.h"

#if SANITIZER_APPLE
#include <libkern/OSAtomic.h>
#include <os/lock.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#if SANITIZER_INTERCEPT_MEMALIGN || SANITIZER_INTERCEPT_PVALLOC
#include <malloc.h>
#endif

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/uio.h>

#if _FILE_OFFSET_BITS == 64 && SANITIZER_GLIBC
const char *const kCreatFunctionName = "creat64";
const char *const kFcntlFunctionName = "fcntl64";
const char *const kFopenFunctionName = "fopen64";
const char *const kOpenAtFunctionName = "openat64";
const char *const kOpenFunctionName = "open64";
const char *const kPreadFunctionName = "pread64";
const char *const kPwriteFunctionName = "pwrite64";
#else
const char *const kCreatFunctionName = "creat";
const char *const kFcntlFunctionName = "fcntl";
const char *const kFopenFunctionName = "fopen";
const char *const kOpenAtFunctionName = "openat";
const char *const kOpenFunctionName = "open";
const char *const kPreadFunctionName = "pread";
const char *const kPwriteFunctionName = "pwrite";
#endif

using namespace testing;
using namespace rtsan_testing;
using namespace std::chrono_literals;

void *FakeThreadEntryPoint(void *) { return nullptr; }

class RtsanFileTest : public ::testing::Test {
protected:
  void SetUp() override {
    const ::testing::TestInfo *const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    file_path_ = std::string("/tmp/rtsan_temporary_test_file_") +
                 test_info->name() + ".txt";
    RemoveTemporaryFile();
  }

  // Gets a file path with the test's name in it
  // This file will be removed if it exists at the end of the test
  const char *GetTemporaryFilePath() const { return file_path_.c_str(); }

  void TearDown() override { RemoveTemporaryFile(); }

private:
  void RemoveTemporaryFile() const { std::remove(GetTemporaryFilePath()); }
  std::string file_path_;
};

/*
    Allocation and deallocation
*/

TEST(TestRtsanInterceptors, MallocDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(nullptr, malloc(1)); };
  ExpectRealtimeDeath(Func, "malloc");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, CallocDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(nullptr, calloc(2, 4)); };
  ExpectRealtimeDeath(Func, "calloc");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, ReallocDiesWhenRealtime) {
  void *ptr_1 = malloc(1);
  auto Func = [ptr_1]() { EXPECT_NE(nullptr, realloc(ptr_1, 8)); };
  ExpectRealtimeDeath(Func, "realloc");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_APPLE
TEST(TestRtsanInterceptors, ReallocfDiesWhenRealtime) {
  void *ptr_1 = malloc(1);
  auto Func = [ptr_1]() { EXPECT_NE(nullptr, reallocf(ptr_1, 8)); };
  ExpectRealtimeDeath(Func, "reallocf");
  ExpectNonRealtimeSurvival(Func);
}
#endif

TEST(TestRtsanInterceptors, VallocDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(nullptr, valloc(4)); };
  ExpectRealtimeDeath(Func, "valloc");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_INTERCEPT_ALIGNED_ALLOC
TEST(TestRtsanInterceptors, AlignedAllocDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(nullptr, aligned_alloc(16, 32)); };
  ExpectRealtimeDeath(Func, "aligned_alloc");
  ExpectNonRealtimeSurvival(Func);
}
#endif

// free_sized and free_aligned_sized (both C23) are not yet supported
TEST(TestRtsanInterceptors, FreeDiesWhenRealtime) {
  void *ptr_1 = malloc(1);
  void *ptr_2 = malloc(1);
  ExpectRealtimeDeath([ptr_1]() { free(ptr_1); }, "free");
  ExpectNonRealtimeSurvival([ptr_2]() { free(ptr_2); });

  // Prevent malloc/free pair being optimised out
  ASSERT_NE(nullptr, ptr_1);
  ASSERT_NE(nullptr, ptr_2);
}

TEST(TestRtsanInterceptors, FreeSurvivesWhenRealtimeIfArgumentIsNull) {
  RealtimeInvoke([]() { free(NULL); });
  ExpectNonRealtimeSurvival([]() { free(NULL); });
}

TEST(TestRtsanInterceptors, PosixMemalignDiesWhenRealtime) {
  auto Func = []() {
    void *ptr;
    posix_memalign(&ptr, 4, 4);
  };
  ExpectRealtimeDeath(Func, "posix_memalign");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_INTERCEPT_MEMALIGN
TEST(TestRtsanInterceptors, MemalignDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(memalign(2, 2048), nullptr); };
  ExpectRealtimeDeath(Func, "memalign");
  ExpectNonRealtimeSurvival(Func);
}
#endif

#if SANITIZER_INTERCEPT_PVALLOC
TEST(TestRtsanInterceptors, PvallocDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(pvalloc(2048), nullptr); };
  ExpectRealtimeDeath(Func, "pvalloc");
  ExpectNonRealtimeSurvival(Func);
}
#endif

/*
    Sleeping
*/

TEST(TestRtsanInterceptors, SleepDiesWhenRealtime) {
  auto Func = []() { sleep(0u); };
  ExpectRealtimeDeath(Func, "sleep");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, UsleepDiesWhenRealtime) {
  auto Func = []() { usleep(1u); };
  ExpectRealtimeDeath(Func, "usleep");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, NanosleepDiesWhenRealtime) {
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

TEST_F(RtsanFileTest, OpenDiesWhenRealtime) {
  auto Func = [this]() { open(GetTemporaryFilePath(), O_RDONLY); };
  ExpectRealtimeDeath(Func, kOpenFunctionName);
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanFileTest, OpenatDiesWhenRealtime) {
  auto Func = [this]() { openat(0, GetTemporaryFilePath(), O_RDONLY); };
  ExpectRealtimeDeath(Func, kOpenAtFunctionName);
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanFileTest, OpenCreatesFileWithProperMode) {
  const mode_t existing_umask = umask(0);
  umask(existing_umask);

  const int mode = S_IRGRP | S_IROTH | S_IRUSR | S_IWUSR;

  const int fd = open(GetTemporaryFilePath(), O_CREAT | O_WRONLY, mode);
  ASSERT_THAT(fd, Ne(-1));
  close(fd);

  struct stat st;
  ASSERT_THAT(stat(GetTemporaryFilePath(), &st), Eq(0));

  // Mask st_mode to get permission bits only
  const mode_t actual_mode = st.st_mode & 0777;
  const mode_t expected_mode = mode & ~existing_umask;
  ASSERT_THAT(actual_mode, Eq(expected_mode));
}

TEST_F(RtsanFileTest, CreatDiesWhenRealtime) {
  auto Func = [this]() { creat(GetTemporaryFilePath(), S_IWOTH | S_IROTH); };
  ExpectRealtimeDeath(Func, kCreatFunctionName);
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, FcntlDiesWhenRealtime) {
  auto Func = []() { fcntl(0, F_GETFL); };
  ExpectRealtimeDeath(Func, kFcntlFunctionName);
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanFileTest, FcntlFlockDiesWhenRealtime) {
  int fd = creat(GetTemporaryFilePath(), S_IRUSR | S_IWUSR);
  ASSERT_THAT(fd, Ne(-1));

  auto Func = [fd]() {
    struct flock lock {};
    lock.l_type = F_RDLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;
    lock.l_pid = ::getpid();

    ASSERT_THAT(fcntl(fd, F_GETLK, &lock), Eq(0));
    ASSERT_THAT(lock.l_type, F_UNLCK);
  };
  ExpectRealtimeDeath(Func, kFcntlFunctionName);
  ExpectNonRealtimeSurvival(Func);

  close(fd);
}

TEST_F(RtsanFileTest, FcntlSetFdDiesWhenRealtime) {
  int fd = creat(GetTemporaryFilePath(), S_IRUSR | S_IWUSR);
  ASSERT_THAT(fd, Ne(-1));

  auto Func = [fd]() {
    int old_flags = fcntl(fd, F_GETFD);
    ASSERT_THAT(fcntl(fd, F_SETFD, FD_CLOEXEC), Eq(0));

    int flags = fcntl(fd, F_GETFD);
    ASSERT_THAT(flags, Ne(-1));
    ASSERT_THAT(flags & FD_CLOEXEC, Eq(FD_CLOEXEC));

    ASSERT_THAT(fcntl(fd, F_SETFD, old_flags), Eq(0));
    ASSERT_THAT(fcntl(fd, F_GETFD), Eq(old_flags));
  };

  ExpectRealtimeDeath(Func, kFcntlFunctionName);
  ExpectNonRealtimeSurvival(Func);

  close(fd);
}

TEST(TestRtsanInterceptors, CloseDiesWhenRealtime) {
  auto Func = []() { close(0); };
  ExpectRealtimeDeath(Func, "close");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanFileTest, FopenDiesWhenRealtime) {
  auto Func = [this]() {
    FILE *f = fopen(GetTemporaryFilePath(), "w");
    EXPECT_THAT(f, Ne(nullptr));
  };

  ExpectRealtimeDeath(Func, kFopenFunctionName);
  ExpectNonRealtimeSurvival(Func);
}

class RtsanOpenedFileTest : public RtsanFileTest {
protected:
  void SetUp() override {
    RtsanFileTest::SetUp();
    file = fopen(GetTemporaryFilePath(), "w");
    ASSERT_THAT(file, Ne(nullptr));
    fd = fileno(file);
    ASSERT_THAT(fd, Ne(-1));
  }

  void TearDown() override {
    if (file != nullptr)
      fclose(file);
    RtsanFileTest::TearDown();
  }

  FILE *GetOpenFile() { return file; }

  int GetOpenFd() { return fd; }

private:
  FILE *file = nullptr;
  int fd = -1;
};

TEST_F(RtsanOpenedFileTest, FreadDiesWhenRealtime) {
  auto Func = [this]() {
    char c{};
    fread(&c, 1, 1, GetOpenFile());
  };
  ExpectRealtimeDeath(Func, "fread");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, FwriteDiesWhenRealtime) {
  const char *message = "Hello, world!";
  auto Func = [&]() { fwrite(&message, 1, 4, GetOpenFile()); };
  ExpectRealtimeDeath(Func, "fwrite");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanFileTest, FcloseDiesWhenRealtime) {
  FILE *f = fopen(GetTemporaryFilePath(), "w");
  EXPECT_THAT(f, Ne(nullptr));
  auto Func = [f]() { fclose(f); };
  ExpectRealtimeDeath(Func, "fclose");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, PutsDiesWhenRealtime) {
  auto Func = []() { puts("Hello, world!\n"); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, FputsDiesWhenRealtime) {
  auto Func = [this]() { fputs("Hello, world!\n", GetOpenFile()); };
  ExpectRealtimeDeath(Func);
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, ReadDiesWhenRealtime) {
  auto Func = [this]() {
    char c{};
    read(GetOpenFd(), &c, 1);
  };
  ExpectRealtimeDeath(Func, "read");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, WriteDiesWhenRealtime) {
  auto Func = [this]() {
    char c = 'a';
    write(GetOpenFd(), &c, 1);
  };
  ExpectRealtimeDeath(Func, "write");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, PreadDiesWhenRealtime) {
  auto Func = [this]() {
    char c{};
    pread(GetOpenFd(), &c, 1, 0);
  };
  ExpectRealtimeDeath(Func, kPreadFunctionName);
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, ReadvDiesWhenRealtime) {
  auto Func = [this]() {
    char c{};
    iovec iov{&c, 1};
    readv(GetOpenFd(), &iov, 1);
  };
  ExpectRealtimeDeath(Func, "readv");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, PwriteDiesWhenRealtime) {
  auto Func = [this]() {
    char c = 'a';
    pwrite(GetOpenFd(), &c, 1, 0);
  };
  ExpectRealtimeDeath(Func, kPwriteFunctionName);
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, WritevDiesWhenRealtime) {
  auto Func = [this]() {
    char c = 'a';
    iovec iov{&c, 1};
    writev(GetOpenFd(), &iov, 1);
  };
  ExpectRealtimeDeath(Func, "writev");
  ExpectNonRealtimeSurvival(Func);
}

/*
    Concurrency
*/

TEST(TestRtsanInterceptors, PthreadCreateDiesWhenRealtime) {
  auto Func = []() {
    pthread_t thread{};
    const pthread_attr_t attr{};
    struct thread_info *thread_info{};
    pthread_create(&thread, &attr, &FakeThreadEntryPoint, thread_info);
  };
  ExpectRealtimeDeath(Func, "pthread_create");
  ExpectNonRealtimeSurvival(Func);
}

class PthreadMutexLockTest : public ::testing::Test {
protected:
  void SetUp() override {
    pthread_mutex_init(&mutex, nullptr);
    is_locked = false;
  }

  void TearDown() override {
    if (is_locked)
      Unlock();

    pthread_mutex_destroy(&mutex);
  }

  void Lock() {
    ASSERT_TRUE(!is_locked);
    pthread_mutex_lock(&mutex);
    is_locked = true;
  }

  void Unlock() {
    ASSERT_TRUE(is_locked);
    pthread_mutex_unlock(&mutex);
    is_locked = false;
  }

private:
  pthread_mutex_t mutex;
  bool is_locked;
};

TEST_F(PthreadMutexLockTest, PthreadMutexLockDiesWhenRealtime) {
  auto Func = [this]() { Lock(); };

  ExpectRealtimeDeath(Func, "pthread_mutex_lock");
}

TEST_F(PthreadMutexLockTest, PthreadMutexLockSurvivesWhenNotRealtime) {
  auto Func = [this]() { Lock(); };

  ExpectNonRealtimeSurvival(Func);
}

TEST_F(PthreadMutexLockTest, PthreadMutexUnlockDiesWhenRealtime) {
  Lock();
  auto Func = [this]() { Unlock(); };

  ExpectRealtimeDeath(Func, "pthread_mutex_unlock");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(PthreadMutexLockTest, PthreadMutexUnlockSurvivesWhenNotRealtime) {
  Lock();
  auto Func = [this]() { Unlock(); };

  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, PthreadJoinDiesWhenRealtime) {
  pthread_t thread{};
  ASSERT_EQ(0,
            pthread_create(&thread, nullptr, &FakeThreadEntryPoint, nullptr));

  auto Func = [&thread]() { pthread_join(thread, nullptr); };

  ExpectRealtimeDeath(Func, "pthread_join");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_APPLE

#pragma clang diagnostic push
// OSSpinLockLock is deprecated, but still in use in libc++
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
TEST(TestRtsanInterceptors, OsSpinLockLockDiesWhenRealtime) {
  auto Func = []() {
    OSSpinLock spin_lock{};
    OSSpinLockLock(&spin_lock);
  };
  ExpectRealtimeDeath(Func, "OSSpinLockLock");
  ExpectNonRealtimeSurvival(Func);
}
#pragma clang diagnostic pop

TEST(TestRtsanInterceptors, OsUnfairLockLockDiesWhenRealtime) {
  auto Func = []() {
    os_unfair_lock_s unfair_lock{};
    os_unfair_lock_lock(&unfair_lock);
  };
  ExpectRealtimeDeath(Func, "os_unfair_lock_lock");
  ExpectNonRealtimeSurvival(Func);
}
#endif

#if SANITIZER_LINUX
TEST(TestRtsanInterceptors, SpinLockLockDiesWhenRealtime) {
  pthread_spinlock_t spin_lock;
  pthread_spin_init(&spin_lock, PTHREAD_PROCESS_SHARED);
  auto Func = [&]() { pthread_spin_lock(&spin_lock); };
  ExpectRealtimeDeath(Func, "pthread_spin_lock");
  ExpectNonRealtimeSurvival(Func);
}
#endif

TEST(TestRtsanInterceptors, PthreadCondSignalDiesWhenRealtime) {
  pthread_cond_t cond{};
  pthread_cond_init(&cond, NULL);

  auto Func = [&cond]() { pthread_cond_signal(&cond); };
  ExpectRealtimeDeath(Func, "pthread_cond_signal");
  ExpectNonRealtimeSurvival(Func);

  pthread_cond_destroy(&cond);
}

TEST(TestRtsanInterceptors, PthreadCondBroadcastDiesWhenRealtime) {
  pthread_cond_t cond{};
  pthread_cond_init(&cond, NULL);

  auto Func = [&cond]() { pthread_cond_broadcast(&cond); };
  ExpectRealtimeDeath(Func, "pthread_cond_broadcast");
  ExpectNonRealtimeSurvival(Func);

  pthread_cond_destroy(&cond);
}

TEST(TestRtsanInterceptors, PthreadCondWaitDiesWhenRealtime) {
  pthread_cond_t cond;
  pthread_mutex_t mutex;
  ASSERT_EQ(0, pthread_cond_init(&cond, nullptr));
  ASSERT_EQ(0, pthread_mutex_init(&mutex, nullptr));

  auto Func = [&]() { pthread_cond_wait(&cond, &mutex); };
  ExpectRealtimeDeath(Func, "pthread_cond_wait");
  // It's very difficult to test the success case here without doing some
  // sleeping, which is at the mercy of the scheduler. What's really important
  // here is the interception - so we're only testing that for now.

  pthread_cond_destroy(&cond);
  pthread_mutex_destroy(&mutex);
}

class PthreadRwlockTest : public ::testing::Test {
protected:
  void SetUp() override {
    pthread_rwlock_init(&rw_lock, nullptr);
    is_locked = false;
  }

  void TearDown() override {
    if (is_locked)
      Unlock();

    pthread_rwlock_destroy(&rw_lock);
  }

  void RdLock() {
    ASSERT_TRUE(!is_locked);
    pthread_rwlock_rdlock(&rw_lock);
    is_locked = true;
  }

  void WrLock() {
    ASSERT_TRUE(!is_locked);
    pthread_rwlock_wrlock(&rw_lock);
    is_locked = true;
  }

  void Unlock() {
    ASSERT_TRUE(is_locked);
    pthread_rwlock_unlock(&rw_lock);
    is_locked = false;
  }

private:
  pthread_rwlock_t rw_lock;
  bool is_locked;
};

TEST_F(PthreadRwlockTest, PthreadRwlockRdlockDiesWhenRealtime) {
  auto Func = [this]() { RdLock(); };
  ExpectRealtimeDeath(Func, "pthread_rwlock_rdlock");
}

TEST_F(PthreadRwlockTest, PthreadRwlockRdlockSurvivesWhenNonRealtime) {
  auto Func = [this]() { RdLock(); };
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(PthreadRwlockTest, PthreadRwlockUnlockDiesWhenRealtime) {
  RdLock();

  auto Func = [this]() { Unlock(); };
  ExpectRealtimeDeath(Func, "pthread_rwlock_unlock");
}

TEST_F(PthreadRwlockTest, PthreadRwlockUnlockSurvivesWhenNonRealtime) {
  RdLock();

  auto Func = [this]() { Unlock(); };
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(PthreadRwlockTest, PthreadRwlockWrlockDiesWhenRealtime) {
  auto Func = [this]() { WrLock(); };

  ExpectRealtimeDeath(Func, "pthread_rwlock_wrlock");
}

TEST_F(PthreadRwlockTest, PthreadRwlockWrlockSurvivesWhenNonRealtime) {
  auto Func = [this]() { WrLock(); };

  ExpectNonRealtimeSurvival(Func);
}

/*
    Sockets
*/
TEST(TestRtsanInterceptors, OpeningASocketDiesWhenRealtime) {
  auto Func = []() { socket(PF_INET, SOCK_STREAM, 0); };
  ExpectRealtimeDeath(Func, "socket");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, SendToASocketDiesWhenRealtime) {
  auto Func = []() { send(0, nullptr, 0, 0); };
  ExpectRealtimeDeath(Func, "send");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, SendmsgToASocketDiesWhenRealtime) {
  msghdr msg{};
  auto Func = [&]() { sendmsg(0, &msg, 0); };
  ExpectRealtimeDeath(Func, "sendmsg");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, SendtoToASocketDiesWhenRealtime) {
  sockaddr addr{};
  socklen_t len{};
  auto Func = [&]() { sendto(0, nullptr, 0, 0, &addr, len); };
  ExpectRealtimeDeath(Func, "sendto");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, RecvFromASocketDiesWhenRealtime) {
  auto Func = []() { recv(0, nullptr, 0, 0); };
  ExpectRealtimeDeath(Func, "recv");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, RecvfromOnASocketDiesWhenRealtime) {
  sockaddr addr{};
  socklen_t len{};
  auto Func = [&]() { recvfrom(0, nullptr, 0, 0, &addr, &len); };
  ExpectRealtimeDeath(Func, "recvfrom");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, RecvmsgOnASocketDiesWhenRealtime) {
  msghdr msg{};
  auto Func = [&]() { recvmsg(0, &msg, 0); };
  ExpectRealtimeDeath(Func, "recvmsg");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, ShutdownOnASocketDiesWhenRealtime) {
  auto Func = [&]() { shutdown(0, 0); };
  ExpectRealtimeDeath(Func, "shutdown");
  ExpectNonRealtimeSurvival(Func);
}
