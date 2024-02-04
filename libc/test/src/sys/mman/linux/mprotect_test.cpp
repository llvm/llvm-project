//===-- Unittests for mprotect --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/mprotect.h"
#include "src/sys/mman/munmap.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <signal.h>
#include <sys/mman.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcMProtectTest, NoError) {
  size_t alloc_size = 128;
  libc_errno = 0;
  void *addr = LIBC_NAMESPACE::mmap(nullptr, alloc_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_NE(addr, MAP_FAILED);

  int *array = reinterpret_cast<int *>(addr);
  // Reading from the memory should not crash the test.
  // Since we used the MAP_ANONYMOUS flag, the contents of the newly
  // allocated memory should be initialized to zero.
  EXPECT_EQ(array[0], 0);

  // By setting the memory protection to read and write, we should be able to
  // modify that memory.
  EXPECT_THAT(
      LIBC_NAMESPACE::mprotect(addr, alloc_size, PROT_READ | PROT_WRITE),
      Succeeds());
  array[0] = 1;
  EXPECT_EQ(array[0], 1);

  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, alloc_size), Succeeds());
}

// This test is disabled currently due to flakeyness. It will be re-enabled once
// it is less flakey.
/*
TEST(LlvmLibcMProtectTest, Error_InvalidWrite) {
  // attempting to write to a read-only protected part of memory should cause a
  // segfault.
  EXPECT_DEATH(
      [] {
        size_t alloc_size = 128;
        void *addr =
            LIBC_NAMESPACE::mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                              MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        LIBC_NAMESPACE::mprotect(addr, alloc_size, PROT_READ);

        (reinterpret_cast<char *>(addr))[0] = 'A';
      },
      WITH_SIGNAL(SIGSEGV));
  // Reading from a write only segment may succeed on some platforms, so there's
  // no test to check that.
}
*/
