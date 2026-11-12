//===-- Unittests for ftok ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fcntl/open.h"
#include "src/sys/ipc/ftok.h"
#include "src/unistd/close.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/fcntl_macros.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcFtokTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcFtokTest, InvalidPath) {
  ASSERT_THAT(LIBC_NAMESPACE::ftok("no/such/path", 1), Fails(ENOENT));
}

TEST_F(LlvmLibcFtokTest, DeterministicForPathAndId) {
  // create a file
  constexpr const char *TEST_FILE_NAME = "ftok.test";
  auto TEST_FILE = libc_make_test_file_path(TEST_FILE_NAME);

  // we first ensure such file exist and set to readable, writable
  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_WRONLY, 0600);
  ASSERT_GT(fd, -1);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));

  // create keys based on file path and user specified int
  key_t key1 = LIBC_NAMESPACE::ftok(TEST_FILE, 'A');
  ASSERT_NE(key1, key_t(-1));
  ASSERT_ERRNO_SUCCESS();

  key_t key2 = LIBC_NAMESPACE::ftok(TEST_FILE, 'A');
  ASSERT_NE(key2, key_t(-1));
  ASSERT_ERRNO_SUCCESS();

  // key should be identical if both inputs are the same
  ASSERT_EQ(key1, key2);

  // create another key
  key_t key3 = LIBC_NAMESPACE::ftok(TEST_FILE, 'B');
  ASSERT_NE(key3, key_t(-1));
  ASSERT_ERRNO_SUCCESS();

  // key should be different if any input is different
  ASSERT_NE(key1, key3);

  // delete the file
  ASSERT_THAT(LIBC_NAMESPACE::unlink(TEST_FILE), Succeeds(0));
}
