//===-- Unittests for mlock -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/errno/libc_errno.h"
#include "src/sys/mman/madvise.h"
#include "src/sys/mman/mincore.h"
#include "src/sys/mman/mlock.h"
#include "src/sys/mman/mlock2.h"
#include "src/sys/mman/mlockall.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munlock.h"
#include "src/sys/mman/munlockall.h"

#include "src/sys/mman/munmap.h"
#include "src/unistd/sysconf.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include "test/UnitTest/Test.h"

#include <asm-generic/errno-base.h>
#include <asm-generic/mman.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

struct PageHolder {
  size_t size;
  void *addr;

  PageHolder()
      : size(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE)),
        addr(LIBC_NAMESPACE::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)) {}
  ~PageHolder() {
    if (addr != MAP_FAILED)
      LIBC_NAMESPACE::munmap(addr, size);
  }

  char &operator[](size_t i) { return reinterpret_cast<char *>(addr)[i]; }

  bool is_valid() { return addr != MAP_FAILED; }
};

TEST(LlvmLibcMlockTest, UnMappedMemory) {
  EXPECT_THAT(LIBC_NAMESPACE::mlock(nullptr, 1024), Fails(ENOMEM));
  EXPECT_THAT(LIBC_NAMESPACE::munlock(nullptr, 1024), Fails(ENOMEM));
}

TEST(LlvmLibcMlockTest, Overflow) {
  PageHolder holder;
  EXPECT_TRUE(holder.is_valid());
  EXPECT_THAT(LIBC_NAMESPACE::mlock(holder.addr, -holder.size), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::munlock(holder.addr, -holder.size),
              Fails(EINVAL));
}

#ifdef SYS_mlock2
TEST(LlvmLibcMlockTest, MLock2) {
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

TEST(LlvmLibcMlockTest, MLockAll) {
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
