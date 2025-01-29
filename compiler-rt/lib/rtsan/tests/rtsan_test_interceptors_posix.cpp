//===--- rtsan_test_interceptors.cpp - Realtime Sanitizer -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_POSIX

#include "gtest/gtest.h"

#include "sanitizer_common/sanitizer_platform_interceptors.h"

#include "rtsan_test_utilities.h"

#if SANITIZER_APPLE
#include <libkern/OSAtomic.h>
#include <os/lock.h>
#include <unistd.h>
#endif

#if SANITIZER_INTERCEPT_MEMALIGN || SANITIZER_INTERCEPT_PVALLOC
#include <malloc.h>
#endif

#if SANITIZER_INTERCEPT_EPOLL
#include <sys/epoll.h>
#endif

#if SANITIZER_INTERCEPT_KQUEUE
#include <sys/event.h>
#include <sys/time.h>
#endif

#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#if SANITIZER_LINUX
#include <sys/inotify.h>
#endif
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/uio.h>

#if _FILE_OFFSET_BITS == 64 && SANITIZER_GLIBC
// Under these conditions, some system calls are `foo64` instead of `foo`
#define MAYBE_APPEND_64(func) func "64"
#else
#define MAYBE_APPEND_64(func) func
#endif

using namespace testing;
using namespace rtsan_testing;
using namespace std::chrono_literals;

// NOTE: In the socket tests we pass in bad info to the calls to ensure they
//       fail which is why we EXPECT_NE 0 for their return codes.
//       We just care that the call is intercepted
const int kNotASocketFd = 0;

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

#if __has_builtin(__builtin_available) && SANITIZER_APPLE
#define ALIGNED_ALLOC_AVAILABLE() (__builtin_available(macOS 10.15, *))
#else
// We are going to assume this is true until we hit systems where it isn't
#define ALIGNED_ALLOC_AVAILABLE() (true)
#endif

TEST(TestRtsanInterceptors, AlignedAllocDiesWhenRealtime) {
  if (ALIGNED_ALLOC_AVAILABLE()) {
    auto Func = []() { EXPECT_NE(nullptr, aligned_alloc(16, 32)); };
    ExpectRealtimeDeath(Func, "aligned_alloc");
    ExpectNonRealtimeSurvival(Func);
  }
}

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

TEST(TestRtsanInterceptors, MmapDiesWhenRealtime) {
  auto Func = []() {
    void *_ = mmap(nullptr, 8, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  };
  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("mmap"));
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_LINUX
TEST(TestRtsanInterceptors, MremapDiesWhenRealtime) {
  void *addr = mmap(nullptr, 8, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  auto Func = [addr]() { void *_ = mremap(addr, 8, 16, 0); };
  ExpectRealtimeDeath(Func, "mremap");
  ExpectNonRealtimeSurvival(Func);
}
#endif

TEST(TestRtsanInterceptors, MunmapDiesWhenRealtime) {
  void *ptr = mmap(nullptr, 8, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  EXPECT_NE(ptr, nullptr);
  auto Func = [ptr]() { munmap(ptr, 8); };
  printf("Right before death munmap\n");
  ExpectRealtimeDeath(Func, "munmap");
  ExpectNonRealtimeSurvival(Func);
}

class RtsanOpenedMmapTest : public RtsanFileTest {
protected:
  void SetUp() override {
    RtsanFileTest::SetUp();
    file = fopen(GetTemporaryFilePath(), "w+");
    ASSERT_THAT(file, Ne(nullptr));
    fd = fileno(file);
    ASSERT_THAT(fd, Ne(-1));
    int ret = ftruncate(GetOpenFd(), size);
    ASSERT_THAT(ret, Ne(-1));
    addr =
        mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, GetOpenFd(), 0);
    ASSERT_THAT(addr, Ne(MAP_FAILED));
    ASSERT_THAT(addr, Ne(nullptr));
  }

  void TearDown() override {
    if (addr != nullptr && addr != MAP_FAILED)
      munmap(addr, size);
    RtsanFileTest::TearDown();
  }

  void *GetAddr() { return addr; }
  static constexpr size_t GetSize() { return size; }

  int GetOpenFd() { return fd; }

private:
  void *addr = nullptr;
  static constexpr size_t size = 4096;
  FILE *file = nullptr;
  int fd = -1;
};

#if !SANITIZER_APPLE
TEST_F(RtsanOpenedMmapTest, MadviseDiesWhenRealtime) {
  auto Func = [this]() { madvise(GetAddr(), GetSize(), MADV_NORMAL); };
  ExpectRealtimeDeath(Func, "madvise");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedMmapTest, PosixMadviseDiesWhenRealtime) {
  auto Func = [this]() {
    posix_madvise(GetAddr(), GetSize(), POSIX_MADV_NORMAL);
  };
  ExpectRealtimeDeath(Func, "posix_madvise");
  ExpectNonRealtimeSurvival(Func);
}
#endif

TEST_F(RtsanOpenedMmapTest, MprotectDiesWhenRealtime) {
  auto Func = [this]() { mprotect(GetAddr(), GetSize(), PROT_READ); };
  ExpectRealtimeDeath(Func, "mprotect");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedMmapTest, MsyncDiesWhenRealtime) {
  auto Func = [this]() { msync(GetAddr(), GetSize(), MS_INVALIDATE); };
  ExpectRealtimeDeath(Func, "msync");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedMmapTest, MincoreDiesWhenRealtime) {
#if SANITIZER_APPLE
  std::vector<char> vec(GetSize() / 1024);
#else
  std::vector<unsigned char> vec(GetSize() / 1024);
#endif
  auto Func = [this, &vec]() { mincore(GetAddr(), GetSize(), vec.data()); };
  ExpectRealtimeDeath(Func, "mincore");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, ShmOpenDiesWhenRealtime) {
  auto Func = []() { shm_open("/rtsan_test_shm", O_CREAT | O_RDWR, 0); };
  ExpectRealtimeDeath(Func, "shm_open");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, ShmUnlinkDiesWhenRealtime) {
  auto Func = []() { shm_unlink("/rtsan_test_shm"); };
  ExpectRealtimeDeath(Func, "shm_unlink");
  ExpectNonRealtimeSurvival(Func);
}

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

TEST(TestRtsanInterceptors, SchedYieldDiesWhenRealtime) {
  auto Func = []() { sched_yield(); };
  ExpectRealtimeDeath(Func, "sched_yield");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_LINUX
TEST(TestRtsanInterceptors, SchedGetaffinityDiesWhenRealtime) {
  cpu_set_t set{};
  auto Func = [&set]() { sched_getaffinity(0, sizeof(set), &set); };
  ExpectRealtimeDeath(Func, "sched_getaffinity");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, SchedSetaffinityDiesWhenRealtime) {
  cpu_set_t set{};
  auto Func = [&set]() { sched_setaffinity(0, sizeof(set), &set); };
  ExpectRealtimeDeath(Func, "sched_setaffinity");
  ExpectNonRealtimeSurvival(Func);
}
#endif

/*
    Filesystem
*/

TEST_F(RtsanFileTest, OpenDiesWhenRealtime) {
  auto Func = [this]() { open(GetTemporaryFilePath(), O_RDONLY); };
  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("open"));
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanFileTest, OpenatDiesWhenRealtime) {
  auto Func = [this]() { openat(0, GetTemporaryFilePath(), O_RDONLY); };
  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("openat"));
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
  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("creat"));
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, FcntlDiesWhenRealtime) {
  auto Func = []() { fcntl(0, F_GETFL); };
  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("fcntl"));
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
  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("fcntl"));
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

  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("fcntl"));
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

  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("fopen"));
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_INTERCEPT_FOPENCOOKIE
TEST_F(RtsanFileTest, FopenCookieDieWhenRealtime) {
  FILE *f = fopen(GetTemporaryFilePath(), "w");
  EXPECT_THAT(f, Ne(nullptr));
  struct fholder {
    FILE *fp;
    size_t read;
  } fh = {f, 0};
  auto CookieRead = [](void *cookie, char *buf, size_t size) {
    fholder *p = reinterpret_cast<fholder *>(cookie);
    p->read = fread(static_cast<void *>(buf), 1, size, p->fp);
    EXPECT_NE(0u, p->read);
  };
  cookie_io_functions_t funcs = {(cookie_read_function_t *)&CookieRead, nullptr,
                                 nullptr, nullptr};
  auto Func = [&fh, &funcs]() {
    FILE *f = fopencookie(&fh, "w", funcs);
    EXPECT_THAT(f, Ne(nullptr));
  };

  ExpectRealtimeDeath(Func, "fopencookie");
  ExpectNonRealtimeSurvival(Func);
}
#endif

#if SANITIZER_INTERCEPT_OPEN_MEMSTREAM
TEST_F(RtsanFileTest, OpenMemstreamDiesWhenRealtime) {
  char *buffer;
  size_t size;
  auto Func = [&buffer, &size]() {
    FILE *f = open_memstream(&buffer, &size);
    EXPECT_THAT(f, Ne(nullptr));
  };

  ExpectRealtimeDeath(Func, "open_memstream");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanFileTest, FmemOpenDiesWhenRealtime) {
  char buffer[1024];
  auto Func = [&buffer]() {
    FILE *f = fmemopen(&buffer, sizeof(buffer), "w");
    EXPECT_THAT(f, Ne(nullptr));
  };

  ExpectRealtimeDeath(Func, "fmemopen");
  ExpectNonRealtimeSurvival(Func);
}
#endif

#if SANITIZER_INTERCEPT_SETVBUF
TEST_F(RtsanFileTest, SetbufDieWhenRealtime) {
  char buffer[BUFSIZ];
  FILE *f = fopen(GetTemporaryFilePath(), "w");
  EXPECT_THAT(f, Ne(nullptr));

  auto Func = [f, &buffer]() { setbuf(f, buffer); };

  ExpectRealtimeDeath(Func, "setbuf");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanFileTest, SetvbufDieWhenRealtime) {
  char buffer[1024];
  size_t size = sizeof(buffer);
  FILE *f = fopen(GetTemporaryFilePath(), "w");
  EXPECT_THAT(f, Ne(nullptr));

  auto Func = [f, &buffer, size]() {
    int r = setvbuf(f, buffer, _IOFBF, size);
    EXPECT_THAT(r, Eq(0));
  };

  ExpectRealtimeDeath(Func, "setvbuf");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanFileTest, SetlinebufDieWhenRealtime) {
  FILE *f = fopen(GetTemporaryFilePath(), "w");
  EXPECT_THAT(f, Ne(nullptr));

  auto Func = [f]() { setlinebuf(f); };

  ExpectRealtimeDeath(Func, "setlinebuf");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanFileTest, SetbufferDieWhenRealtime) {
  char buffer[1024];
  size_t size = sizeof(buffer);
  FILE *f = fopen(GetTemporaryFilePath(), "w");
  EXPECT_THAT(f, Ne(nullptr));

  auto Func = [f, &buffer, size]() { setbuffer(f, buffer, size); };

  ExpectRealtimeDeath(Func, "setbuffer");
  ExpectNonRealtimeSurvival(Func);
}
#endif

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

#if SANITIZER_INTERCEPT_FSEEK
TEST_F(RtsanOpenedFileTest, FgetposDieWhenRealtime) {
  auto Func = [this]() {
    fpos_t pos;
    int ret = fgetpos(GetOpenFile(), &pos);
    ASSERT_THAT(ret, Eq(0));
  };

  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("fgetpos"));
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, FsetposDieWhenRealtime) {
  fpos_t pos;
  int ret = fgetpos(GetOpenFile(), &pos);
  ASSERT_THAT(ret, Eq(0));
  auto Func = [this, pos]() {
    int ret = fsetpos(GetOpenFile(), &pos);
    ASSERT_THAT(ret, Eq(0));
  };

  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("fsetpos"));
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, FseekDieWhenRealtime) {
  auto Func = [this]() {
    int ret = fseek(GetOpenFile(), 0, SEEK_CUR);
    ASSERT_THAT(ret, Eq(0));
  };

  ExpectRealtimeDeath(Func, "fseek");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, FseekoDieWhenRealtime) {
  auto Func = [this]() {
    int ret = fseeko(GetOpenFile(), 0, SEEK_CUR);
    ASSERT_THAT(ret, Eq(0));
  };

  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("fseeko"));
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, FtellDieWhenRealtime) {
  auto Func = [this]() {
    long ret = ftell(GetOpenFile());
    ASSERT_THAT(ret, Eq(0));
  };

  ExpectRealtimeDeath(Func, "ftell");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, FtelloDieWhenRealtime) {
  auto Func = [this]() {
    off_t ret = ftello(GetOpenFile());
    ASSERT_THAT(ret, Eq(0));
  };

  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("ftello"));
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, RewindDieWhenRealtime) {
  int end = fseek(GetOpenFile(), 0, SEEK_END);
  EXPECT_THAT(end, Eq(0));
  auto Func = [this]() { rewind(GetOpenFile()); };

  ExpectRealtimeDeath(Func, "rewind");
  ExpectNonRealtimeSurvival(Func);
}
#endif

TEST(TestRtsanInterceptors, IoctlDiesWhenRealtime) {
  auto Func = []() { ioctl(0, FIONREAD); };
  ExpectRealtimeDeath(Func, "ioctl");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, IoctlBehavesWithOutputArg) {
  int arg{};
  ioctl(GetOpenFd(), FIONREAD, &arg);

  EXPECT_THAT(arg, Ge(0));
}

TEST_F(RtsanOpenedFileTest, FdopenDiesWhenRealtime) {
  auto Func = [&]() {
    FILE *f = fdopen(GetOpenFd(), "w");
    EXPECT_THAT(f, Ne(nullptr));
  };

  ExpectRealtimeDeath(Func, "fdopen");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, FreopenDiesWhenRealtime) {
  auto Func = [&]() {
    FILE *newfile = freopen(GetTemporaryFilePath(), "w", GetOpenFile());
    EXPECT_THAT(newfile, Ne(nullptr));
  };

  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("freopen"));
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, IoctlBehavesWithOutputPointer) {
  // These initial checks just see if we CAN run these tests.
  // If we can't (can't open a socket, or can't find an interface, just
  // gracefully skip.
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock == -1) {
    perror("socket");
    GTEST_SKIP();
  }

  struct ifaddrs *ifaddr = nullptr;
  if (getifaddrs(&ifaddr) == -1 || ifaddr == nullptr) {
    perror("getifaddrs");
    close(sock);
    GTEST_SKIP();
  }

  struct ifreq ifr {};
  strncpy(ifr.ifr_name, ifaddr->ifa_name, IFNAMSIZ - 1);

  int retval = ioctl(sock, SIOCGIFADDR, &ifr);
  if (retval == -1) {
    perror("ioctl");
    close(sock);
    freeifaddrs(ifaddr);
    FAIL();
  }

  freeifaddrs(ifaddr);
  close(sock);

  ASSERT_THAT(ifr.ifr_addr.sa_data, NotNull());
  ASSERT_THAT(ifr.ifr_addr.sa_family, Eq(AF_INET));
}

TEST_F(RtsanOpenedFileTest, LseekDiesWhenRealtime) {
  auto Func = [this]() { lseek(GetOpenFd(), 0, SEEK_SET); };
  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("lseek"));
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, DupDiesWhenRealtime) {
  auto Func = [this]() { dup(GetOpenFd()); };
  ExpectRealtimeDeath(Func, "dup");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, Dup2DiesWhenRealtime) {
  auto Func = [this]() { dup2(GetOpenFd(), 0); };
  ExpectRealtimeDeath(Func, "dup2");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanFileTest, ChmodDiesWhenRealtime) {
  auto Func = [this]() { chmod(GetTemporaryFilePath(), 0777); };
  ExpectRealtimeDeath(Func, "chmod");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanOpenedFileTest, FchmodDiesWhenRealtime) {
  auto Func = [this]() { fchmod(GetOpenFd(), 0777); };
  ExpectRealtimeDeath(Func, "fchmod");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, UmaskDiesWhenRealtime) {
  auto Func = []() { umask(0); };
  ExpectRealtimeDeath(Func, "umask");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_INTERCEPT_PROCESS_VM_READV
TEST(TestRtsanInterceptors, ProcessVmReadvDiesWhenRealtime) {
  char stack[1024];
  int p;
  iovec lcl{&stack, sizeof(stack)};
  iovec rmt{&p, sizeof(p)};
  auto Func = [&lcl, &rmt]() { process_vm_readv(0, &lcl, 1, &rmt, 1, 0); };
  ExpectRealtimeDeath(Func, "process_vm_readv");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, ProcessVmWritevDiesWhenRealtime) {
  char stack[1024];
  int p;
  iovec lcl{&p, sizeof(p)};
  iovec rmt{&stack, sizeof(stack)};
  auto Func = [&lcl, &rmt]() { process_vm_writev(0, &lcl, 1, &rmt, 1, 0); };
  ExpectRealtimeDeath(Func, "process_vm_writev");
  ExpectNonRealtimeSurvival(Func);
}
#endif

class RtsanDirectoryTest : public ::testing::Test {
protected:
  void SetUp() override {
    const ::testing::TestInfo *const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    directory_path_ = std::string("/tmp/rtsan_temp_dir_") + test_info->name();
    RemoveTemporaryDirectory();
  }

  const char *GetTemporaryDirectoryPath() const {
    return directory_path_.c_str();
  }

  void TearDown() override { RemoveTemporaryDirectory(); }

private:
  void RemoveTemporaryDirectory() const {
    std::remove(GetTemporaryDirectoryPath());
  }
  std::string directory_path_;
};

TEST_F(RtsanDirectoryTest, MkdirDiesWhenRealtime) {
  auto Func = [this]() { mkdir(GetTemporaryDirectoryPath(), 0777); };
  ExpectRealtimeDeath(Func, "mkdir");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(RtsanDirectoryTest, RmdirDiesWhenRealtime) {
  // We don't actually create this directory before we try to remove it
  // Thats OK - we are just making sure the call gets intercepted
  auto Func = [this]() { rmdir(GetTemporaryDirectoryPath()); };
  ExpectRealtimeDeath(Func, "rmdir");
  ExpectNonRealtimeSurvival(Func);
}

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

TEST_F(RtsanFileTest, FflushDiesWhenRealtime) {
  FILE *f = fopen(GetTemporaryFilePath(), "w");
  EXPECT_THAT(f, Ne(nullptr));
  int written = fwrite("abc", 1, 3, f);
  EXPECT_THAT(written, Eq(3));
  auto Func = [&f]() {
    int res = fflush(f);
    EXPECT_THAT(res, Eq(0));
  };
  ExpectRealtimeDeath(Func, "fflush");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_APPLE
TEST_F(RtsanFileTest, FpurgeDiesWhenRealtime) {
  FILE *f = fopen(GetTemporaryFilePath(), "w");
  EXPECT_THAT(f, Ne(nullptr));
  int written = fwrite("abc", 1, 3, f);
  EXPECT_THAT(written, Eq(3));
  auto Func = [&f]() {
    int res = fpurge(f);
    EXPECT_THAT(res, Eq(0));
  };
  ExpectRealtimeDeath(Func, "fpurge");
  ExpectNonRealtimeSurvival(Func);
}
#endif

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
  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("pread"));
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_INTERCEPT_PREADV
TEST_F(RtsanOpenedFileTest, PreadvDiesWhenRealtime) {
  auto Func = [this]() {
    char c{};
    iovec iov{&c, sizeof(c)};
    preadv(GetOpenFd(), &iov, 1, 0);
  };
  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("preadv"));
  ExpectNonRealtimeSurvival(Func);
}
#endif

#if SANITIZER_INTERCEPT_PWRITEV
TEST_F(RtsanOpenedFileTest, PwritevDiesWhenRealtime) {
  auto Func = [this]() {
    char c{};
    iovec iov{&c, sizeof(c)};
    pwritev(GetOpenFd(), &iov, 1, 0);
  };
  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("pwritev"));
  ExpectNonRealtimeSurvival(Func);
}
#endif

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
  ExpectRealtimeDeath(Func, MAYBE_APPEND_64("pwrite"));
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
  ASSERT_EQ(0, pthread_cond_init(&cond, nullptr));

  auto Func = [&cond]() { pthread_cond_signal(&cond); };
  ExpectRealtimeDeath(Func, "pthread_cond_signal");
  ExpectNonRealtimeSurvival(Func);

  pthread_cond_destroy(&cond);
}

TEST(TestRtsanInterceptors, PthreadCondBroadcastDiesWhenRealtime) {
  pthread_cond_t cond{};
  ASSERT_EQ(0, pthread_cond_init(&cond, nullptr));

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
TEST(TestRtsanInterceptors, GetAddrInfoDiesWhenRealtime) {
  auto Func = []() {
    addrinfo *info{};
    getaddrinfo("localhost", "http", nullptr, &info);
  };
  ExpectRealtimeDeath(Func, "getaddrinfo");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, GetNameInfoDiesWhenRealtime) {
  auto Func = []() {
    char host[NI_MAXHOST];
    char serv[NI_MAXSERV];
    getnameinfo(nullptr, 0, host, NI_MAXHOST, serv, NI_MAXSERV, 0);
  };
  ExpectRealtimeDeath(Func, "getnameinfo");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, BindingASocketDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(bind(kNotASocketFd, nullptr, 0), 0); };
  ExpectRealtimeDeath(Func, "bind");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, ListeningOnASocketDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(listen(kNotASocketFd, 0), 0); };
  ExpectRealtimeDeath(Func, "listen");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, AcceptingASocketDiesWhenRealtime) {
  auto Func = []() { EXPECT_LT(accept(kNotASocketFd, nullptr, nullptr), 0); };
  ExpectRealtimeDeath(Func, "accept");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_INTERCEPT_ACCEPT4
TEST(TestRtsanInterceptors, Accepting4ASocketDiesWhenRealtime) {
  auto Func = []() {
    EXPECT_LT(accept4(kNotASocketFd, nullptr, nullptr, 0), 0);
  };
  ExpectRealtimeDeath(Func, "accept4");
  ExpectNonRealtimeSurvival(Func);
}
#endif

TEST(TestRtsanInterceptors, ConnectingASocketDiesWhenRealtime) {
  auto Func = []() { EXPECT_NE(connect(kNotASocketFd, nullptr, 0), 0); };
  ExpectRealtimeDeath(Func, "connect");
  ExpectNonRealtimeSurvival(Func);
}

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

#if SANITIZER_INTERCEPT_SENDMMSG
TEST(TestRtsanInterceptors, SendmmsgOnASocketDiesWhenRealtime) {
  mmsghdr msg{};
  auto Func = [&]() { sendmmsg(0, &msg, 0, 0); };
  ExpectRealtimeDeath(Func, "sendmmsg");
  ExpectNonRealtimeSurvival(Func);
}
#endif

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

#if SANITIZER_INTERCEPT_RECVMMSG
TEST(TestRtsanInterceptors, RecvmmsgOnASocketDiesWhenRealtime) {
  mmsghdr msg{};
  auto Func = [&]() { recvmmsg(0, &msg, 0, 0, nullptr); };
  ExpectRealtimeDeath(Func, "recvmmsg");
  ExpectNonRealtimeSurvival(Func);
}
#endif

TEST(TestRtsanInterceptors, ShutdownOnASocketDiesWhenRealtime) {
  auto Func = [&]() { shutdown(0, 0); };
  ExpectRealtimeDeath(Func, "shutdown");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_INTERCEPT_GETSOCKNAME
TEST(TestRtsanInterceptors, GetsocknameOnASocketDiesWhenRealtime) {
  sockaddr addr{};
  socklen_t len{};
  auto Func = [&]() { getsockname(0, &addr, &len); };
  ExpectRealtimeDeath(Func, "getsockname");
  ExpectNonRealtimeSurvival(Func);
}
#endif

#if SANITIZER_INTERCEPT_GETPEERNAME
TEST(TestRtsanInterceptors, GetpeernameOnASocketDiesWhenRealtime) {
  sockaddr addr{};
  socklen_t len{};
  auto Func = [&]() { getpeername(0, &addr, &len); };
  ExpectRealtimeDeath(Func, "getpeername");
  ExpectNonRealtimeSurvival(Func);
}
#endif

#if SANITIZER_INTERCEPT_GETSOCKOPT
TEST(TestRtsanInterceptors, GetsockoptOnASocketDiesWhenRealtime) {
  int val = 0;
  socklen_t len = static_cast<socklen_t>(sizeof(val));
  auto Func = [&val, &len]() {
    getsockopt(0, SOL_SOCKET, SO_REUSEADDR, &val, &len);
  };
  ExpectRealtimeDeath(Func, "getsockopt");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, SetsockoptOnASocketDiesWhenRealtime) {
  int val = 0;
  socklen_t len = static_cast<socklen_t>(sizeof(val));
  auto Func = [&val, &len]() {
    setsockopt(0, SOL_SOCKET, SO_REUSEADDR, &val, len);
  };
  ExpectRealtimeDeath(Func, "setsockopt");
  ExpectNonRealtimeSurvival(Func);
}
#endif

TEST(TestRtsanInterceptors, SocketpairDiesWhenRealtime) {
  int pair[2]{};
  auto Func = [&pair]() { socketpair(0, 0, 0, pair); };
  ExpectRealtimeDeath(Func, "socketpair");
  ExpectNonRealtimeSurvival(Func);
}

/*
    I/O Multiplexing
*/

TEST(TestRtsanInterceptors, PollDiesWhenRealtime) {
  struct pollfd fds[1];
  fds[0].fd = 0;
  fds[0].events = POLLIN;

  auto Func = [&fds]() { poll(fds, 1, 0); };

  ExpectRealtimeDeath(Func, "poll");
  ExpectNonRealtimeSurvival(Func);
}

#if !SANITIZER_APPLE
// FIXME: This should work on Darwin as well
// see the comment near the interceptor
TEST(TestRtsanInterceptors, SelectDiesWhenRealtime) {
  fd_set readfds;
  FD_ZERO(&readfds);
  FD_SET(0, &readfds);
  struct timeval timeout = {0, 0};

  auto Func = [&readfds, &timeout]() {
    select(1, &readfds, nullptr, nullptr, &timeout);
  };
  ExpectRealtimeDeath(Func, "select");
  ExpectNonRealtimeSurvival(Func);
}
#endif

TEST(TestRtsanInterceptors, PSelectDiesWhenRealtime) {
  fd_set readfds;
  FD_ZERO(&readfds);
  FD_SET(0, &readfds);
  struct timespec timeout = {0, 0};

  auto Func = [&]() {
    pselect(1, &readfds, nullptr, nullptr, &timeout, nullptr);
  };
  ExpectRealtimeDeath(Func, "pselect");
  ExpectNonRealtimeSurvival(Func);
}

#if SANITIZER_INTERCEPT_EPOLL
TEST(TestRtsanInterceptors, EpollCreateDiesWhenRealtime) {
  auto Func = []() { epoll_create(1); };
  ExpectRealtimeDeath(Func, "epoll_create");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, EpollCreate1DiesWhenRealtime) {
  auto Func = []() { epoll_create1(EPOLL_CLOEXEC); };
  ExpectRealtimeDeath(Func, "epoll_create1");
  ExpectNonRealtimeSurvival(Func);
}

class EpollTest : public ::testing::Test {
protected:
  void SetUp() override {
    epfd = epoll_create1(EPOLL_CLOEXEC);
    ASSERT_GE(epfd, 0);
  }

  void TearDown() override {
    if (epfd >= 0)
      close(epfd);
  }

  int GetEpollFd() { return epfd; }

private:
  int epfd = -1;
};

TEST_F(EpollTest, EpollCtlDiesWhenRealtime) {
  auto Func = [this]() {
    struct epoll_event event = {.events = EPOLLIN, .data = {.fd = 0}};
    epoll_ctl(GetEpollFd(), EPOLL_CTL_ADD, 0, &event);
  };
  ExpectRealtimeDeath(Func, "epoll_ctl");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(EpollTest, EpollWaitDiesWhenRealtime) {
  auto Func = [this]() {
    struct epoll_event events[1];
    epoll_wait(GetEpollFd(), events, 1, 0);
  };

  ExpectRealtimeDeath(Func, "epoll_wait");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(EpollTest, EpollPWaitDiesWhenRealtime) {
  auto Func = [this]() {
    struct epoll_event events[1];
    epoll_pwait(GetEpollFd(), events, 1, 0, nullptr);
  };

  ExpectRealtimeDeath(Func, "epoll_pwait");
  ExpectNonRealtimeSurvival(Func);
}
#endif // SANITIZER_INTERCEPT_EPOLL

#if SANITIZER_INTERCEPT_PPOLL
TEST(TestRtsanInterceptors, PpollDiesWhenRealtime) {
  struct pollfd fds[1];
  fds[0].fd = 0;
  fds[0].events = POLLIN;

  timespec ts = {0, 0};

  auto Func = [&fds, &ts]() { ppoll(fds, 1, &ts, nullptr); };

  ExpectRealtimeDeath(Func, "ppoll");
  ExpectNonRealtimeSurvival(Func);
}
#endif

#if SANITIZER_INTERCEPT_KQUEUE
TEST(TestRtsanInterceptors, KqueueDiesWhenRealtime) {
  auto Func = []() { kqueue(); };
  ExpectRealtimeDeath(Func, "kqueue");
  ExpectNonRealtimeSurvival(Func);
}

class KqueueTest : public ::testing::Test {
protected:
  void SetUp() override {
    kq = kqueue();
    ASSERT_GE(kq, 0);
  }

  void TearDown() override {
    if (kq >= 0)
      close(kq);
  }

  int GetKqueueFd() { return kq; }

private:
  int kq = -1;
};

TEST_F(KqueueTest, KeventDiesWhenRealtime) {
  struct kevent event;
  EV_SET(&event, 0, EVFILT_READ, EV_ADD, 0, 0, nullptr);
  struct timespec timeout = {0, 0};

  auto Func = [this, event, timeout]() {
    kevent(GetKqueueFd(), &event, 1, nullptr, 0, &timeout);
  };

  ExpectRealtimeDeath(Func, "kevent");
  ExpectNonRealtimeSurvival(Func);
}

TEST_F(KqueueTest, Kevent64DiesWhenRealtime) {
  struct kevent64_s event;
  EV_SET64(&event, 0, EVFILT_READ, EV_ADD, 0, 0, 0, 0, 0);
  struct timespec timeout = {0, 0};

  auto Func = [this, event, timeout]() {
    kevent64(GetKqueueFd(), &event, 1, nullptr, 0, 0, &timeout);
  };

  ExpectRealtimeDeath(Func, "kevent64");
  ExpectNonRealtimeSurvival(Func);
}
#endif // SANITIZER_INTERCEPT_KQUEUE

#if SANITIZER_LINUX
TEST(TestRtsanInterceptors, InotifyInitDiesWhenRealtime) {
  auto Func = []() { inotify_init(); };
  ExpectRealtimeDeath(Func, "inotify_init");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, InotifyInit1DiesWhenRealtime) {
  auto Func = []() { inotify_init1(0); };
  ExpectRealtimeDeath(Func, "inotify_init1");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, InotifyAddWatchDiesWhenRealtime) {
  int fd = inotify_init();
  EXPECT_THAT(fd, Ne(-1));
  auto Func = [fd]() {
    inotify_add_watch(fd, "/tmp/rtsan_inotify", IN_CREATE);
  };
  ExpectRealtimeDeath(Func, "inotify_add_watch");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, InotifyRmWatchDiesWhenRealtime) {
  int fd = inotify_init();
  EXPECT_THAT(fd, Ne(-1));
  auto Func = [fd]() { inotify_rm_watch(fd, -1); };
  ExpectRealtimeDeath(Func, "inotify_rm_watch");
  ExpectNonRealtimeSurvival(Func);
}
#endif

TEST(TestRtsanInterceptors, MkfifoDiesWhenRealtime) {
  auto Func = []() { mkfifo("/tmp/rtsan_test_fifo", 0); };
  ExpectRealtimeDeath(Func, "mkfifo");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, PipeDiesWhenRealtime) {
  int fds[2];
  auto Func = [&fds]() { pipe(fds); };
  ExpectRealtimeDeath(Func, "pipe");
  ExpectNonRealtimeSurvival(Func);
}

#if !SANITIZER_APPLE
TEST(TestRtsanInterceptors, Pipe2DiesWhenRealtime) {
  int fds[2];
  auto Func = [&fds]() { pipe2(fds, O_CLOEXEC); };
  ExpectRealtimeDeath(Func, "pipe2");
  ExpectNonRealtimeSurvival(Func);
}
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
TEST(TestRtsanInterceptors, SyscallDiesWhenRealtime) {
  auto Func = []() { syscall(SYS_getpid); };
  ExpectRealtimeDeath(Func, "syscall");
  ExpectNonRealtimeSurvival(Func);
}

TEST(TestRtsanInterceptors, GetPidReturnsSame) {
  int pid = syscall(SYS_getpid);
  EXPECT_THAT(pid, Ne(-1));

  EXPECT_THAT(getpid(), Eq(pid));
}
#pragma clang diagnostic pop

#endif // SANITIZER_POSIX
