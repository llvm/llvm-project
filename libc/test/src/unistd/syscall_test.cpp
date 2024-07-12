//===-- Unittests for syscalls --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/unistd/syscall.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <fcntl.h>
#include <sys/stat.h>    // For S_* flags.
#include <sys/syscall.h> // For syscall numbers.
#include <unistd.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

// We only do a smoke test here. Actual functionality tests are
// done by the unit tests of the syscall wrappers like mmap.
// The goal is to test syscalls with a wide number of args.

// There is no function named "syscall" in llvm-libc, we instead use a macro to
// set up the arguments properly. We still need to specify the namespace though
// because the macro generates a call to the actual internal function
// (__llvm_libc_syscall) which is inside the namespace.
TEST(LlvmLibcSyscallTest, TrivialCall) {
  LIBC_NAMESPACE::libc_errno = 0;

  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_gettid), 0l);
  ASSERT_ERRNO_SUCCESS();
}

TEST(LlvmLibcSyscallTest, SymlinkCreateDestroy) {
  constexpr const char LINK_VAL[] = "syscall_readlink_test_value";
  constexpr const char LINK[] = "testdata/syscall_readlink.test.link";

#ifdef SYS_symlink
  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_symlink, LINK_VAL, LINK), 0l);
#elif defined(SYS_symlinkat)
  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_symlinkat, LINK_VAL, AT_FDCWD, LINK),
            0l);
#else
#error "symlink and symlinkat syscalls not available."
#endif
  ASSERT_ERRNO_SUCCESS();

  char buf[sizeof(LINK_VAL)];

#ifdef SYS_readlink
  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_readlink, LINK, buf, sizeof(buf)), 0l);
#elif defined(SYS_readlinkat)
  ASSERT_GE(
      LIBC_NAMESPACE::syscall(SYS_readlinkat, AT_FDCWD, LINK, buf, sizeof(buf)),
      0l);
#endif
  ASSERT_ERRNO_SUCCESS();

#ifdef SYS_unlink
  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_unlink, LINK), 0l);
#elif defined(SYS_unlinkat)
  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_unlinkat, AT_FDCWD, LINK, 0), 0l);
#else
#error "unlink and unlinkat syscalls not available."
#endif
  ASSERT_ERRNO_SUCCESS();
}

TEST(LlvmLibcSyscallTest, FileReadWrite) {
  constexpr const char HELLO[] = "hello";
  constexpr int HELLO_SIZE = sizeof(HELLO);

  constexpr const char *TEST_FILE = "testdata/syscall_pread_pwrite.test";

#ifdef SYS_open
  int fd =
      LIBC_NAMESPACE::syscall(SYS_open, TEST_FILE, O_WRONLY | O_CREAT, S_IRWXU);
#elif defined(SYS_openat)
  int fd = LIBC_NAMESPACE::syscall(SYS_openat, AT_FDCWD, TEST_FILE,
                                   O_WRONLY | O_CREAT, S_IRWXU);
#else
#error "open and openat syscalls not available."
#endif
  ASSERT_GT(fd, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_pwrite64, fd, HELLO, HELLO_SIZE, 0),
            0l);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_fsync, fd), 0l);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_close, fd), 0l);
  ASSERT_ERRNO_SUCCESS();
}

TEST(LlvmLibcSyscallTest, FileLinkCreateDestroy) {
  constexpr const char *TEST_DIR = "testdata";
  constexpr const char *TEST_FILE = "syscall_linkat.test";
  constexpr const char *TEST_FILE_PATH = "testdata/syscall_linkat.test";
  constexpr const char *TEST_FILE_LINK = "syscall_linkat.test.link";
  constexpr const char *TEST_FILE_LINK_PATH =
      "testdata/syscall_linkat.test.link";

  // The test strategy is as follows:
  //   1. Create a normal file
  //   2. Create a link to that file.
  //   3. Open the link to check that the link was created.
  //   4. Cleanup the file and its link.

#ifdef SYS_open
  int write_fd = LIBC_NAMESPACE::syscall(SYS_open, TEST_FILE_PATH,
                                         O_WRONLY | O_CREAT, S_IRWXU);
#elif defined(SYS_openat)
  int write_fd = LIBC_NAMESPACE::syscall(SYS_openat, AT_FDCWD, TEST_FILE_PATH,
                                         O_WRONLY | O_CREAT, S_IRWXU);
#else
#error "open and openat syscalls not available."
#endif
  ASSERT_GT(write_fd, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_close, write_fd), 0l);
  ASSERT_ERRNO_SUCCESS();

#ifdef SYS_open
  int dir_fd = LIBC_NAMESPACE::syscall(SYS_open, TEST_DIR, O_DIRECTORY, 0);
#elif defined(SYS_openat)
  int dir_fd =
      LIBC_NAMESPACE::syscall(SYS_openat, AT_FDCWD, TEST_DIR, O_DIRECTORY, 0);
#else
#error "open and openat syscalls not available."
#endif
  ASSERT_GT(dir_fd, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_linkat, dir_fd, TEST_FILE, dir_fd,
                                    TEST_FILE_LINK, 0),
            0l);
  ASSERT_ERRNO_SUCCESS();
#ifdef SYS_open
  int link_fd =
      LIBC_NAMESPACE::syscall(SYS_open, TEST_FILE_LINK_PATH, O_PATH, 0);
#elif defined(SYS_openat)
  int link_fd = LIBC_NAMESPACE::syscall(SYS_openat, AT_FDCWD,
                                        TEST_FILE_LINK_PATH, O_PATH, 0);
#else
#error "open and openat syscalls not available."
#endif
  ASSERT_GT(link_fd, 0);
  ASSERT_ERRNO_SUCCESS();

#ifdef SYS_unlink
  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_unlink, TEST_FILE_PATH), 0l);
#elif defined(SYS_unlinkat)
  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_unlinkat, AT_FDCWD, TEST_FILE_PATH, 0),
            0l);
#else
#error "unlink and unlinkat syscalls not available."
#endif
  ASSERT_ERRNO_SUCCESS();

#ifdef SYS_unlink
  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_unlink, TEST_FILE_LINK_PATH), 0l);
#elif defined(SYS_unlinkat)
  ASSERT_GE(
      LIBC_NAMESPACE::syscall(SYS_unlinkat, AT_FDCWD, TEST_FILE_LINK_PATH, 0),
      0l);
#else
#error "unlink and unlinkat syscalls not available."
#endif
  ASSERT_ERRNO_SUCCESS();

  ASSERT_GE(LIBC_NAMESPACE::syscall(SYS_close, dir_fd), 0l);
  ASSERT_ERRNO_SUCCESS();
}
