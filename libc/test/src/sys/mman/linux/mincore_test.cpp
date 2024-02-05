//===-- Unittests for mincore ---------------------------------------------===//
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
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munlock.h"
#include "src/sys/mman/munmap.h"
#include "src/unistd/sysconf.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include "test/UnitTest/Test.h"

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcMincoreTest, UnMappedMemory) {
  libc_errno = 0;
  unsigned char vec;
  int res = LIBC_NAMESPACE::mincore(nullptr, 1, &vec);
  EXPECT_THAT(res, Fails(ENOMEM, -1));
}

TEST(LlvmLibcMincoreTest, UnalignedAddr) {
  unsigned long page_size = LIBC_NAMESPACE::sysconf(_SC_PAGESIZE);
  void *addr = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % page_size, 0ul);
  libc_errno = 0;
  int res = LIBC_NAMESPACE::mincore(static_cast<char *>(addr) + 1, 1, nullptr);
  EXPECT_THAT(res, Fails(EINVAL, -1));
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}

TEST(LlvmLibcMincoreTest, InvalidVec) {
  unsigned long page_size = LIBC_NAMESPACE::sysconf(_SC_PAGESIZE);
  void *addr = LIBC_NAMESPACE::mmap(nullptr, 4 * page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % page_size, 0ul);
  libc_errno = 0;
  int res = LIBC_NAMESPACE::mincore(addr, 1, nullptr);
  EXPECT_THAT(res, Fails(EFAULT, -1));
}

TEST(LlvmLibcMincoreTest, NoError) {
  unsigned long page_size = LIBC_NAMESPACE::sysconf(_SC_PAGESIZE);
  void *addr = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % page_size, 0ul);
  unsigned char vec;
  libc_errno = 0;
  int res = LIBC_NAMESPACE::mincore(addr, 1, &vec);
  EXPECT_THAT(res, Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}

TEST(LlvmLibcMincoreTest, NegativeLength) {
  unsigned long page_size = LIBC_NAMESPACE::sysconf(_SC_PAGESIZE);
  void *addr = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % page_size, 0ul);
  unsigned char vec;
  libc_errno = 0;
  int res = LIBC_NAMESPACE::mincore(addr, -1, &vec);
  EXPECT_THAT(res, Fails(ENOMEM, -1));
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}

TEST(LlvmLibcMincoreTest, PageOut) {
  unsigned long page_size = LIBC_NAMESPACE::sysconf(_SC_PAGESIZE);
  unsigned char vec;
  void *addr = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ | PROT_WRITE,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % page_size, 0ul);

  // touch the page
  {
    static_cast<char *>(addr)[0] = 0;
    EXPECT_THAT(LIBC_NAMESPACE::mlock(addr, page_size), Succeeds());
    int res = LIBC_NAMESPACE::mincore(addr, 1, &vec);
    EXPECT_EQ(vec & 1u, 1u);
    EXPECT_THAT(res, Succeeds());
    EXPECT_THAT(LIBC_NAMESPACE::munlock(addr, page_size), Succeeds());
  }

  // page out the memory
  {
    libc_errno = 0;
    EXPECT_THAT(LIBC_NAMESPACE::madvise(addr, page_size, MADV_DONTNEED),
                Succeeds());

    libc_errno = 0;
    int res = LIBC_NAMESPACE::mincore(addr, page_size, &vec);
    EXPECT_EQ(vec & 1u, 0u);
    EXPECT_THAT(res, Succeeds());
  }

  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}
