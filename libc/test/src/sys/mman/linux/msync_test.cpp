//===-- Unittests for msync -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/mman/mlock.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/msync.h"
#include "src/sys/mman/munlock.h"
#include "src/sys/mman/munmap.h"
#include "src/unistd/sysconf.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

const size_t PAGE_SIZE = LIBC_NAMESPACE::sysconf(_SC_PAGESIZE);

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcMsyncTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

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

TEST_F(LlvmLibcMsyncTest, UnMappedMemory) {
  EXPECT_THAT(LIBC_NAMESPACE::msync(nullptr, 1024, MS_SYNC), Fails(ENOMEM));
  EXPECT_THAT(LIBC_NAMESPACE::msync(nullptr, 1024, MS_ASYNC), Fails(ENOMEM));
}

TEST_F(LlvmLibcMsyncTest, LockedPage) {
  PageHolder page;
  ASSERT_TRUE(page.is_valid());
  ASSERT_THAT(LIBC_NAMESPACE::mlock(page.addr, page.size), Succeeds());
  EXPECT_THAT(
      LIBC_NAMESPACE::msync(page.addr, page.size, MS_SYNC | MS_INVALIDATE),
      Fails(EBUSY));
  ASSERT_THAT(LIBC_NAMESPACE::munlock(page.addr, page.size), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::msync(page.addr, page.size, MS_SYNC), Succeeds());
}

TEST_F(LlvmLibcMsyncTest, UnalignedAddress) {
  PageHolder page;
  ASSERT_TRUE(page.is_valid());
  EXPECT_THAT(LIBC_NAMESPACE::msync(&page[1], page.size - 1, MS_SYNC),
              Fails(EINVAL));
}

TEST_F(LlvmLibcMsyncTest, InvalidFlag) {
  PageHolder page;
  ASSERT_TRUE(page.is_valid());
  EXPECT_THAT(LIBC_NAMESPACE::msync(page.addr, page.size, MS_SYNC | MS_ASYNC),
              Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::msync(page.addr, page.size, -1), Fails(EINVAL));
}
