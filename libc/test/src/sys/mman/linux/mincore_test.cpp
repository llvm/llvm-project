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
#include "src/unistd/sysconf.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include "test/UnitTest/Test.h"

#include <sys/mman.h>
#include <unistd.h> // For sysconf.

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcMincoreTest, UnMappedMemory) {
  libc_errno = 0;
  unsigned char vec;
  int res = LIBC_NAMESPACE::mincore(nullptr, 1, &vec);
  EXPECT_THAT(res, Fails(ENOMEM, -1));
}

TEST(LlvmLibcMincoreTest, InvalidVec) {
  size_t page_size = static_cast<size_t>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE));
  void *addr = LIBC_NAMESPACE::mmap(nullptr, 4 * page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % page_size, 0ul);
  libc_errno = 0;
  int res = LIBC_NAMESPACE::mincore(addr, 1, nullptr);
  EXPECT_THAT(res, Fails(EFAULT, -1));
  void *area = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ | PROT_WRITE,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(area, MAP_FAILED);
  unsigned char *ptr = static_cast<unsigned char *>(area) + page_size - 3;
  res = LIBC_NAMESPACE::mincore(addr, 4 * page_size, ptr);
  EXPECT_THAT(res, Fails(EFAULT, -1));
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::munmap(area, 2), Succeeds());
}

TEST(LlvmLibcMincoreTest, UnalignedAddr) {
  size_t page_size = static_cast<size_t>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE));
  void *addr = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % page_size, 0ul);
  libc_errno = 0;
  int res = LIBC_NAMESPACE::mincore(static_cast<char *>(addr) + 1, 1, nullptr);
  EXPECT_THAT(res, Fails(EINVAL, -1));
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}

TEST(LlvmLibcMincoreTest, NoError) {
  size_t page_size = static_cast<size_t>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE));
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
  size_t page_size = static_cast<size_t>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE));
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
  unsigned char vec;
  size_t page_size = static_cast<size_t>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE));
  void *addr = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ | PROT_WRITE,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % page_size, 0ul);

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
    EXPECT_THAT(LIBC_NAMESPACE::madvise(addr, page_size, MADV_DONTNEED),
                Succeeds());

    libc_errno = 0;
    int res = LIBC_NAMESPACE::mincore(addr, page_size, &vec);
    EXPECT_EQ(vec & 1u, 0u);
    EXPECT_THAT(res, Succeeds());
  }

  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}
