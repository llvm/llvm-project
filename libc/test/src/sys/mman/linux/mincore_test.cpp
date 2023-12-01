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

// It is always possible to find an aligned boundary if we allocate page sized
// memory.
static char *aligned_addr(void *addr, size_t alignment) {
  char *byte_addr = static_cast<char *>(addr);
  uintptr_t addr_val = reinterpret_cast<uintptr_t>(addr);
  uintptr_t offset =
      addr_val % alignment == 0 ? 0 : alignment - (addr_val % alignment);
  return byte_addr + offset;
}

TEST(LlvmLibcMincoreTest, InvalidVec) {
  size_t page_size = static_cast<size_t>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE));
  void *addr = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  char *aligned = aligned_addr(addr, page_size);
  int res = LIBC_NAMESPACE::mincore(aligned, 1, nullptr);
  EXPECT_THAT(res, Fails(EFAULT, -1));
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}

TEST(LlvmLibcMincoreTest, UnalignedAddr) {
  size_t page_size = static_cast<size_t>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE));
  void *addr = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  char *aligned = aligned_addr(addr, page_size);
  libc_errno = 0;
  int res = LIBC_NAMESPACE::mincore(aligned + 1, 1, nullptr);
  EXPECT_THAT(res, Fails(EINVAL, -1));
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}

TEST(LlvmLibcMincoreTest, NoError) {
  size_t page_size = static_cast<size_t>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE));
  void *addr = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  char *aligned = aligned_addr(addr, page_size);
  unsigned char vec;
  libc_errno = 0;
  int res = LIBC_NAMESPACE::mincore(aligned, 1, &vec);
  EXPECT_THAT(res, Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}

TEST(LlvmLibcMincoreTest, NegativeLength) {
  size_t page_size = static_cast<size_t>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE));
  void *addr = LIBC_NAMESPACE::mmap(nullptr, page_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  char *aligned = aligned_addr(addr, page_size);
  unsigned char vec;
  libc_errno = 0;
  int res = LIBC_NAMESPACE::mincore(aligned, -1, &vec);
  EXPECT_THAT(res, Fails(ENOMEM, -1));
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, page_size), Succeeds());
}

TEST(LlvmLibcMincoreTest, PageOut) {
  unsigned char vec;
  size_t page_size = static_cast<size_t>(LIBC_NAMESPACE::sysconf(_SC_PAGESIZE));
  // allocate 2 pages since we need to page out page_size bytes
  void *addr =
      LIBC_NAMESPACE::mmap(nullptr, 2 * page_size, PROT_READ | PROT_WRITE,
                           MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_NE(addr, MAP_FAILED);
  char *aligned = aligned_addr(addr, page_size);

  // touch the page
  {
    aligned[0] = 0;
    libc_errno = 0;
    int res = LIBC_NAMESPACE::mincore(aligned, 1, &vec);
    EXPECT_EQ(vec & 1u, 1u);
    EXPECT_THAT(res, Succeeds());
  }

  // page out the memory
  {
    libc_errno = 0;
    EXPECT_THAT(LIBC_NAMESPACE::madvise(aligned, page_size, MADV_DONTNEED),
                Succeeds());

    libc_errno = 0;
    int res = LIBC_NAMESPACE::mincore(aligned, 1, &vec);
    EXPECT_EQ(vec & 1u, 0u);
    EXPECT_THAT(res, Succeeds());
  }

  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, 2 * page_size), Succeeds());
}
