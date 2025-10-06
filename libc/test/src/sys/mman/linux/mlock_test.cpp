//===-- Unittests for mlock -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: Simplify these tests and split them up. mlock, mlock2, mlockall,
// munlock, and munlockall should have separate test files which only need to
// check our code paths (succeeds and errors).

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/libc_errno.h"
#include "src/sys/mman/madvise.h"
#include "src/sys/mman/mincore.h"
#include "src/sys/mman/mlock.h"
#include "src/sys/mman/mlock2.h"
#include "src/sys/mman/mlockall.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munlock.h"
#include "src/sys/mman/munlockall.h"
#include "src/sys/mman/munmap.h"
#include "src/sys/resource/getrlimit.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/syscall.h>

// TODO: Replace with sysconf call once the function is properly implemented.
constexpr size_t PAGE_SIZE = 4096;

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcMlockTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

struct PageHolder {
  size_t size;
  void *addr;

  PageHolder()
      : size(PAGE_SIZE),
        addr(LIBC_NAMESPACE::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) {}
  ~PageHolder() {
    if (addr != MAP_FAILED)
      LIBC_NAMESPACE::munmap(addr, size);
  }

  char &operator[](size_t i) { return reinterpret_cast<char *>(addr)[i]; }

  bool is_valid() { return addr != MAP_FAILED; }
};

static bool is_permitted_size(size_t size) {
  rlimit rlimits;
  LIBC_NAMESPACE::getrlimit(RLIMIT_MEMLOCK, &rlimits);
  return size <= static_cast<size_t>(rlimits.rlim_cur);
}

TEST_F(LlvmLibcMlockTest, UnMappedMemory) {
  EXPECT_THAT(LIBC_NAMESPACE::mlock(nullptr, 1024), Fails(ENOMEM));
  EXPECT_THAT(LIBC_NAMESPACE::munlock(nullptr, 1024), Fails(ENOMEM));
}

TEST_F(LlvmLibcMlockTest, Overflow) {
  PageHolder holder;
  EXPECT_TRUE(holder.is_valid());
  size_t negative_size = -holder.size;
  int expected_errno = is_permitted_size(negative_size) ? EINVAL : ENOMEM;
  EXPECT_THAT(LIBC_NAMESPACE::mlock(holder.addr, negative_size),
              Fails(expected_errno));
  EXPECT_THAT(LIBC_NAMESPACE::munlock(holder.addr, negative_size),
              Fails(EINVAL));
}

#ifdef SYS_mlock2
TEST_F(LlvmLibcMlockTest, MLock2) {
  PageHolder holder;
  EXPECT_TRUE(holder.is_valid());
  EXPECT_THAT(LIBC_NAMESPACE::madvise(holder.addr, holder.size, MADV_DONTNEED),
              Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::mlock2(holder.addr, holder.size, 0), Succeeds());
  unsigned char vec;
  EXPECT_THAT(LIBC_NAMESPACE::mincore(holder.addr, holder.size, &vec),
              Succeeds());
  EXPECT_EQ(vec & 1, 1);
  EXPECT_THAT(LIBC_NAMESPACE::munlock(holder.addr, holder.size), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::madvise(holder.addr, holder.size, MADV_DONTNEED),
              Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::mlock2(holder.addr, holder.size, MLOCK_ONFAULT),
              Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::mincore(holder.addr, holder.size, &vec),
              Succeeds());
  EXPECT_EQ(vec & 1, 0);
  holder[0] = 1;
  EXPECT_THAT(LIBC_NAMESPACE::mincore(holder.addr, holder.size, &vec),
              Succeeds());
  EXPECT_EQ(vec & 1, 1);
  EXPECT_THAT(LIBC_NAMESPACE::munlock(holder.addr, holder.size), Succeeds());
}
#endif

TEST_F(LlvmLibcMlockTest, InvalidFlag) {
  size_t alloc_size = 128; // page size
  void *addr = LIBC_NAMESPACE::mmap(nullptr, alloc_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_NE(addr, MAP_FAILED);

  // Invalid mlock2 flags.
  EXPECT_THAT(LIBC_NAMESPACE::mlock2(addr, alloc_size, 1234), Fails(EINVAL));

  // Invalid mlockall flags.
  EXPECT_THAT(LIBC_NAMESPACE::mlockall(1234), Fails(EINVAL));

  // man 2 mlockall says EINVAL is a valid return code when MCL_ONFAULT was
  // specified without MCL_FUTURE or MCL_CURRENT, but this seems to fail on
  // Linux 4.19.y (EOL).
  // TODO(ndesaulniers) re-enable after
  // https://github.com/llvm/llvm-project/issues/80073 is fixed.
  // EXPECT_THAT(LIBC_NAMESPACE::mlockall(MCL_ONFAULT), Fails(EINVAL));

  LIBC_NAMESPACE::munmap(addr, alloc_size);
}

TEST_F(LlvmLibcMlockTest, MLockAll) {
  {
    PageHolder holder;
    EXPECT_TRUE(holder.is_valid());
    EXPECT_THAT(
        LIBC_NAMESPACE::madvise(holder.addr, holder.size, MADV_DONTNEED),
        Succeeds());
    auto retval = LIBC_NAMESPACE::mlockall(MCL_CURRENT);
    if (retval == -1) {
      EXPECT_TRUE(libc_errno == ENOMEM || libc_errno == EPERM);
      libc_errno = 0;
      return;
    }
    unsigned char vec;
    EXPECT_THAT(LIBC_NAMESPACE::mincore(holder.addr, holder.size, &vec),
                Succeeds());
    EXPECT_EQ(vec & 1, 1);
    EXPECT_THAT(LIBC_NAMESPACE::munlockall(), Succeeds());
  }
  {
    auto retval = LIBC_NAMESPACE::mlockall(MCL_FUTURE);
    if (retval == -1) {
      EXPECT_TRUE(libc_errno == ENOMEM || libc_errno == EPERM);
      libc_errno = 0;
      return;
    }
    PageHolder holder;
    EXPECT_TRUE(holder.is_valid());
    unsigned char vec;
    EXPECT_THAT(LIBC_NAMESPACE::mincore(holder.addr, holder.size, &vec),
                Succeeds());
    EXPECT_EQ(vec & 1, 1);
    EXPECT_THAT(LIBC_NAMESPACE::munlockall(), Succeeds());
  }
#ifdef MCL_ONFAULT
  {
    auto retval = LIBC_NAMESPACE::mlockall(MCL_FUTURE | MCL_ONFAULT);
    if (retval == -1) {
      EXPECT_TRUE(libc_errno == ENOMEM || libc_errno == EPERM);
      libc_errno = 0;
      return;
    }
    PageHolder holder;
    EXPECT_TRUE(holder.is_valid());
    unsigned char vec;
    EXPECT_THAT(LIBC_NAMESPACE::mincore(holder.addr, holder.size, &vec),
                Succeeds());
    EXPECT_EQ(vec & 1, 0);
    holder[0] = 1;
    EXPECT_THAT(LIBC_NAMESPACE::mincore(holder.addr, holder.size, &vec),
                Succeeds());
    EXPECT_EQ(vec & 1, 1);
    EXPECT_THAT(LIBC_NAMESPACE::munlockall(), Succeeds());
  }
#endif
}
