//===-- Unittests for mmap and munmap -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/sys_mman_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/sys/mman/memfd_create.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcMMapTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcMMapTest, NoError) {
  size_t alloc_size = 128;
  void *addr = LIBC_NAMESPACE::mmap(nullptr, alloc_size, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_NE(addr, MAP_FAILED);

  int *array = reinterpret_cast<int *>(addr);
  // Reading from the memory should not crash the test.
  // Since we used the MAP_ANONYMOUS flag, the contents of the newly
  // allocated memory should be initialized to zero.
  EXPECT_EQ(array[0], 0);
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, alloc_size), Succeeds());
}

TEST_F(LlvmLibcMMapTest, Error_InvalidSize) {
  void *addr = LIBC_NAMESPACE::mmap(nullptr, 0, PROT_READ,
                                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  EXPECT_THAT(addr, Fails(EINVAL, MAP_FAILED));

  EXPECT_THAT(LIBC_NAMESPACE::munmap(0, 0), Fails(EINVAL));
}

TEST_F(LlvmLibcMMapTest, FileOffsets) {
  int fd = LIBC_NAMESPACE::memfd_create("mmap_test", MFD_CLOEXEC);
  ASSERT_NE(fd, -1);
  ASSERT_ERRNO_SUCCESS();
  LIBC_NAMESPACE::cpp::scope_exit close_fd(
      [&]() { ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds()); });

  // Check that we can map a file from offset zero. This succeeds even though
  // the file is empty.
  void *addr = LIBC_NAMESPACE::mmap(nullptr, 47, PROT_READ, MAP_PRIVATE, fd, 0);
  ASSERT_NE(addr, MAP_FAILED);
  ASSERT_ERRNO_SUCCESS();
  EXPECT_THAT(LIBC_NAMESPACE::munmap(addr, 47), Succeeds());

  // Mapping negative offsets fails.
  EXPECT_THAT(LIBC_NAMESPACE::mmap(nullptr, 47, PROT_READ, MAP_PRIVATE, fd, -1),
              Fails(EINVAL, MAP_FAILED));

  // So do offsets that are not page aligned. This should be rejected in the
  // kernel or by our mmap2 rounding code. Note that POSIX permits (but does not
  // require) mapping at unaligned offsets, but linux does not support it.
  EXPECT_THAT(LIBC_NAMESPACE::mmap(nullptr, 47, PROT_READ, MAP_PRIVATE, fd, 47),
              Fails(EINVAL, MAP_FAILED));

  if constexpr (sizeof(off_t) > sizeof(long)) {
    // On 32-bit systems, we need to reject offsets that don't fit into syscall
    // arguments, even after the mmap2 shift.
    EXPECT_THAT(
        LIBC_NAMESPACE::mmap(nullptr, 47, PROT_READ, MAP_PRIVATE, fd,
                             static_cast<off_t>(1) << (sizeof(off_t) * 8 - 2)),
        Fails(EINVAL, MAP_FAILED));
  }
}
