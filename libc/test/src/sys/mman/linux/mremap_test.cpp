//===-- Unittests for mremap ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/mremap.h"
#include "src/sys/mman/munmap.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/mman.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcMremapTest, NoError) {
  size_t initial_size = 128;
  size_t new_size = 256;
  LIBC_NAMESPACE::libc_errno = 0;

  // Allocate memory using mmap.
  void *addr =
      LIBC_NAMESPACE::mmap(nullptr, initial_size, PROT_READ | PROT_WRITE,
                           MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_NE(addr, MAP_FAILED);

  int *array = reinterpret_cast<int *>(addr);
  // Writing to the memory should not crash the test.
  array[0] = 123;
  EXPECT_EQ(array[0], 123);

  // Re-map the memory using mremap with an increased size.
  void *new_addr =
      LIBC_NAMESPACE::mremap(addr, initial_size, new_size, MREMAP_MAYMOVE);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_NE(new_addr, MAP_FAILED);
  EXPECT_EQ(reinterpret_cast<int *>(new_addr)[0],
            123); // Verify data is preserved.

  // Clean up memory by unmapping it.
  EXPECT_THAT(LIBC_NAMESPACE::munmap(new_addr, new_size), Succeeds());
}

TEST(LlvmLibcMremapTest, Error_InvalidSize) {
  size_t initial_size = 128;
  LIBC_NAMESPACE::libc_errno = 0;

  // Allocate memory using mmap.
  void *addr =
      LIBC_NAMESPACE::mmap(nullptr, initial_size, PROT_READ | PROT_WRITE,
                           MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_NE(addr, MAP_FAILED);

  // Attempt to re-map the memory with an invalid new size (0).
  void *new_addr =
      LIBC_NAMESPACE::mremap(addr, initial_size, 0, MREMAP_MAYMOVE);
  EXPECT_THAT(new_addr, Fails(EINVAL, MAP_FAILED));

  // Clean up the original mapping.
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, initial_size), Succeeds());
}
