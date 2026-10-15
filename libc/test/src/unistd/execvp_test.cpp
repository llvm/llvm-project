//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for execvp.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/unistd/execvp.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcExecvpTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcExecvpTest, NonExistentFile) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  char *const argv[] = {const_cast<char *>("nonexistent_binary_xyz_123"),
                        nullptr};
  ASSERT_THAT(LIBC_NAMESPACE::execvp("nonexistent_binary_xyz_123", argv),
              Fails(ENOENT));
}

TEST_F(LlvmLibcExecvpTest, NonExistentPathWithSlash) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  char *const argv[] = {const_cast<char *>("nonexistent"), nullptr};
  ASSERT_THAT(LIBC_NAMESPACE::execvp("/nonexistent/path/xyz", argv),
              Fails(ENOENT));
}

TEST_F(LlvmLibcExecvpTest, EmptyFileName) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  char *const argv[] = {const_cast<char *>(""), nullptr};
  ASSERT_THAT(LIBC_NAMESPACE::execvp("", argv), Fails(ENOENT));
}
