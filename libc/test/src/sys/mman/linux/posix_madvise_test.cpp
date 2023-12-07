//===-- Unittests for posix_madvise ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/sys/mman/posix_madvise.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/mman.h>

using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcPosixMadviseTest, NoError) {
  size_t alloc_size = 128;
  libc_errno = 0;
  void *addr = __llvm_libc::mmap(nullptr, alloc_size, PROT_READ,
                                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_EQ(0, libc_errno);
  EXPECT_NE(addr, MAP_FAILED);

  EXPECT_EQ(__llvm_libc::posix_madvise(addr, alloc_size, POSIX_MADV_RANDOM), 0);

  int *array = reinterpret_cast<int *>(addr);
  // Reading from the memory should not crash the test.
  // Since we used the MAP_ANONYMOUS flag, the contents of the newly
  // allocated memory should be initialized to zero.
  EXPECT_EQ(array[0], 0);
  EXPECT_THAT(__llvm_libc::munmap(addr, alloc_size), Succeeds());
}

TEST(LlvmLibcPosixMadviseTest, Error_BadPtr) {
  libc_errno = 0;
  // posix_madvise is a no-op on DONTNEED, so it shouldn't fail even with the
  // nullptr.
  EXPECT_EQ(__llvm_libc::posix_madvise(nullptr, 8, POSIX_MADV_DONTNEED), 0);

  // posix_madvise doesn't set errno, but the return value is actually the error
  // code.
  EXPECT_EQ(__llvm_libc::posix_madvise(nullptr, 8, POSIX_MADV_SEQUENTIAL),
            ENOMEM);
}
