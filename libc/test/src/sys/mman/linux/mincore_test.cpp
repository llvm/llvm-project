//===-- Unittests for mincore ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/madvise.h"
#include "src/sys/mman/mincore.h"
#include "src/sys/mman/mlock.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munlock.h"
#include "src/sys/mman/munmap.h"
#include "src/unistd/sysconf.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcMincoreTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

const size_t PAGE_SIZE = LIBC_NAMESPACE::sysconf(_SC_PAGESIZE);

TEST_F(LlvmLibcMincoreTest, UnMappedMemory) {
  unsigned char vec;
  int res = LIBC_NAMESPACE::mincore(nullptr, 1, &vec);
  EXPECT_THAT(res, Fails(ENOMEM, -1));
}

TEST_F(LlvmLibcMincoreTest, UnalignedAddr) {
  unsigned long page_size = PAGE_SIZE;
  void *addr = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % page_size, 0ul);
  int res = LIBC_NAMESPACE::mincore(static_cast<char *>(addr) + 1, 1, nullptr);
  EXPECT_THAT(res, Fails(EINVAL, -1));
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}

TEST_F(LlvmLibcMincoreTest, InvalidVec) {
  unsigned long page_size = PAGE_SIZE;
  void *addr = LIBC_NAMESPACE::mmap(nullptr, 4 * page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % page_size, 0ul);
  int res = LIBC_NAMESPACE::mincore(addr, 1, nullptr);
  EXPECT_THAT(res, Fails(EFAULT, -1));
}

TEST_F(LlvmLibcMincoreTest, NoError) {
  unsigned long page_size = PAGE_SIZE;
  void *addr = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % page_size, 0ul);
  unsigned char vec;
  int res = LIBC_NAMESPACE::mincore(addr, 1, &vec);
  EXPECT_THAT(res, Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}

TEST_F(LlvmLibcMincoreTest, NegativeLength) {
  unsigned long page_size = PAGE_SIZE;
  void *addr = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<unsigned long>(addr) % page_size, 0ul);
  unsigned char vec;
  int res = LIBC_NAMESPACE::mincore(addr, -1, &vec);
  EXPECT_THAT(res, Fails(ENOMEM, -1));
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}

TEST_F(LlvmLibcMincoreTest, PageOut) {
  unsigned long page_size = PAGE_SIZE;
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
    EXPECT_THAT(LIBC_NAMESPACE::madvise(addr, page_size, MADV_DONTNEED),
                Succeeds());

    int res = LIBC_NAMESPACE::mincore(addr, page_size, &vec);
    EXPECT_EQ(vec & 1u, 0u);
    EXPECT_THAT(res, Succeeds());
  }

  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}
