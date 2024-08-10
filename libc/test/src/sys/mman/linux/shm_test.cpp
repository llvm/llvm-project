//===-- Unittests for shm_open/shm_unlink ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h"
#include "src/fcntl/fcntl.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/sys/mman/shm_open.h"
#include "src/sys/mman/shm_unlink.h"
#include "src/unistd/close.h"
#include "src/unistd/ftruncate.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include <asm-generic/fcntl.h>
#include <sys/syscall.h>

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
// since shm_open/shm_unlink are wrappers around open/unlink, we only focus on
// testing basic cases and name conversions.

TEST(LlvmLibcShmTest, Basic) {
  const char *name = "/test_shm_open";
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::shm_open(name, O_CREAT | O_RDWR, 0666),
              returns(GE(0)).with_errno(EQ(0)));

  // check that FD_CLOEXEC is set by default.
  long flag = LIBC_NAMESPACE::fcntl(fd, F_GETFD);
  ASSERT_GE(static_cast<int>(flag), 0);
  EXPECT_NE(static_cast<int>(flag) & FD_CLOEXEC, 0);

  // allocate space using ftruncate
  ASSERT_THAT(LIBC_NAMESPACE::ftruncate(fd, 4096), Succeeds());
  // map the shared memory
  void *addr = LIBC_NAMESPACE::mmap(nullptr, 4096, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, 0);
  ASSERT_NE(addr, MAP_FAILED);
  // just write random data to the shared memory
  char data[] = "Despite its name, LLVM has little to do with traditional "
                "virtual machines.";
  for (size_t i = 0; i < sizeof(data); ++i)
    static_cast<char *>(addr)[i] = data[i];

  // close fd does not affect the mapping
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds());
  for (size_t i = 0; i < sizeof(data); ++i)
    EXPECT_EQ(static_cast<char *>(addr)[i], data[i]);

  // unmap the shared memory
  ASSERT_THAT(LIBC_NAMESPACE::munmap(addr, 4096), Succeeds());
  // remove the shared memory
  ASSERT_THAT(LIBC_NAMESPACE::shm_unlink(name), Succeeds());
}

TEST(LlvmLibcShmTest, NameConversion) {
  const char *name = "////test_shm_open";
  int fd;
  ASSERT_THAT(fd = LIBC_NAMESPACE::shm_open(name, O_CREAT | O_RDWR, 0666),
              returns(GE(0)).with_errno(EQ(0)));
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::shm_unlink(name), Succeeds());

  ASSERT_THAT(LIBC_NAMESPACE::shm_open("/123/123", O_CREAT | O_RDWR, 0666),
              Fails(EINVAL));

  ASSERT_THAT(LIBC_NAMESPACE::shm_open("/.", O_CREAT | O_RDWR, 0666),
              Fails(EINVAL));

  ASSERT_THAT(LIBC_NAMESPACE::shm_open("/..", O_CREAT | O_RDWR, 0666),
              Fails(EINVAL));
}
