//===-- Unittests for mlock -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/sys/mman/mlock.h"
#include "src/sys/mman/mlock2.h"
#include "src/sys/mman/mlockall.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munlock.h"
#include "src/sys/mman/munlockall.h"
#include "src/sys/mman/munmap.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/mman.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcMlockTest, NoError) {
  size_t alloc_size = 128;
  libc_errno = 0;
  void *addr = LIBC_NAMESPACE::mmap(nullptr, alloc_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_EQ(0, libc_errno);
  EXPECT_NE(addr, MAP_FAILED);

  EXPECT_THAT(LIBC_NAMESPACE::mlock(addr, alloc_size), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::munlock(addr, alloc_size), Succeeds());

  EXPECT_THAT(LIBC_NAMESPACE::mlock2(addr, alloc_size, MLOCK_ONFAULT),
              Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::munlock(addr, alloc_size), Succeeds());

  EXPECT_THAT(LIBC_NAMESPACE::mlockall(MCL_CURRENT), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::munlockall(), Succeeds());

  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, alloc_size), Succeeds());
}

TEST(LlvmLibcMlockTest, NoMem) {
  size_t alloc_size = 4096; // page size
  libc_errno = 0;
  void *addr = LIBC_NAMESPACE::mmap(nullptr, alloc_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_EQ(0, libc_errno);
  EXPECT_NE(addr, MAP_FAILED);
  LIBC_NAMESPACE::munmap(addr, alloc_size);

  // addr holds the address of an unmapped page. Calling mlock with such an
  // address should get EINVAL.
  EXPECT_THAT(LIBC_NAMESPACE::mlock(addr, alloc_size), Fails(ENOMEM));
}

TEST(LlvmLibcMlockTest, InvalidFlag) {
  size_t alloc_size = 128; // page size
  libc_errno = 0;
  void *addr = LIBC_NAMESPACE::mmap(nullptr, alloc_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_EQ(0, libc_errno);
  EXPECT_NE(addr, MAP_FAILED);

  // Invalid mlock2 flags.
  EXPECT_THAT(LIBC_NAMESPACE::mlock2(addr, alloc_size, 1234), Fails(EINVAL));

  // Invalid mlockall flags.
  EXPECT_THAT(LIBC_NAMESPACE::mlockall(1234), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::mlockall(MCL_ONFAULT), Fails(EINVAL));

  LIBC_NAMESPACE::munmap(addr, alloc_size);
}
