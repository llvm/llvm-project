//===-- Unittests for mincore ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/sys/mman/madvise.h"
#include "src/sys/mman/mincore.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include "test/UnitTest/Test.h"

#include <linux/param.h> // For EXEC_PAGESIZE
#include <sys/mman.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcMincoreTest, UnMappedMemory) {
  libc_errno = 0;
  unsigned char vec;
  int res = LIBC_NAMESPACE::mincore(nullptr, 1, &vec);
  EXPECT_THAT(res, Fails(ENOMEM, -1));
}

TEST(LlvmLibcMincoreTest, InvalidVec) {
  void *addr = LIBC_NAMESPACE::mmap(nullptr, 4 * EXEC_PAGESIZE, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % EXEC_PAGESIZE, 0ul);
  libc_errno = 0;
  int res = LIBC_NAMESPACE::mincore(addr, 1, nullptr);
  EXPECT_THAT(res, Fails(EFAULT, -1));
  void *area =
      LIBC_NAMESPACE::mmap(nullptr, EXEC_PAGESIZE, PROT_READ | PROT_WRITE,
                           MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(area, MAP_FAILED);
  unsigned char *ptr = static_cast<unsigned char *>(area) + EXEC_PAGESIZE - 3;
  res = LIBC_NAMESPACE::mincore(addr, 4 * EXEC_PAGESIZE, ptr);
  EXPECT_THAT(res, Fails(EFAULT, -1));
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, EXEC_PAGESIZE), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::munmap(area, 2), Succeeds());
}

TEST(LlvmLibcMincoreTest, UnalignedAddr) {
  void *addr = LIBC_NAMESPACE::mmap(nullptr, EXEC_PAGESIZE, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % EXEC_PAGESIZE, 0ul);
  libc_errno = 0;
  int res = LIBC_NAMESPACE::mincore(static_cast<char *>(addr) + 1, 1, nullptr);
  EXPECT_THAT(res, Fails(EINVAL, -1));
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, EXEC_PAGESIZE), Succeeds());
}

TEST(LlvmLibcMincoreTest, NoError) {
  void *addr = LIBC_NAMESPACE::mmap(nullptr, EXEC_PAGESIZE, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % EXEC_PAGESIZE, 0ul);
  unsigned char vec;
  libc_errno = 0;
  int res = LIBC_NAMESPACE::mincore(addr, 1, &vec);
  EXPECT_THAT(res, Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, EXEC_PAGESIZE), Succeeds());
}

TEST(LlvmLibcMincoreTest, NegativeLength) {
  void *addr = LIBC_NAMESPACE::mmap(nullptr, EXEC_PAGESIZE, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % EXEC_PAGESIZE, 0ul);
  unsigned char vec;
  libc_errno = 0;
  int res = LIBC_NAMESPACE::mincore(addr, -1, &vec);
  EXPECT_THAT(res, Fails(ENOMEM, -1));
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, EXEC_PAGESIZE), Succeeds());
}

TEST(LlvmLibcMincoreTest, PageOut) {
  unsigned char vec;
  void *addr =
      LIBC_NAMESPACE::mmap(nullptr, EXEC_PAGESIZE, PROT_READ | PROT_WRITE,
                           MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % EXEC_PAGESIZE, 0ul);

  // touch the page
  {
    static_cast<char *>(addr)[0] = 0;
    libc_errno = 0;
    int res = LIBC_NAMESPACE::mincore(addr, 1, &vec);
    EXPECT_EQ(vec & 1u, 1u);
    EXPECT_THAT(res, Succeeds());
  }

  // page out the memory
  {
    libc_errno = 0;
    EXPECT_THAT(LIBC_NAMESPACE::madvise(addr, EXEC_PAGESIZE, MADV_DONTNEED),
                Succeeds());

    libc_errno = 0;
    int res = LIBC_NAMESPACE::mincore(addr, EXEC_PAGESIZE, &vec);
    EXPECT_EQ(vec & 1u, 0u);
    EXPECT_THAT(res, Succeeds());
  }

  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, EXEC_PAGESIZE), Succeeds());
}
