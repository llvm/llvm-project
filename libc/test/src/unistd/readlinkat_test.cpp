//===-- Unittests for readlinkat ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/errno/libc_errno.h"
#include "src/string/string_utils.h"
#include "src/unistd/readlinkat.h"
#include "src/unistd/symlink.h"
#include "src/unistd/unlink.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/fcntl_macros.h"

namespace cpp = LIBC_NAMESPACE::cpp;

TEST(LlvmLibcReadlinkatTest, CreateAndUnlink) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  constexpr const char *FILENAME = "readlinkat_test_file";
  auto LINK_VAL = libc_make_test_file_path(FILENAME);
  constexpr const char *FILENAME2 = "readlinkat_test_file.link";
  auto LINK = libc_make_test_file_path(FILENAME2);
  LIBC_NAMESPACE::libc_errno = 0;

  // The test strategy is as follows:
  //   1. Create a symlink with value LINK_VAL.
  //   2. Read the symlink with readlink. The link value read should be LINK_VAL
  //   3. Cleanup the symlink created in step #1.
  ASSERT_THAT(LIBC_NAMESPACE::symlink(LINK_VAL, LINK), Succeeds(0));

  char buf[256];
  ssize_t len = LIBC_NAMESPACE::readlinkat(
      AT_FDCWD, LINK, buf, LIBC_NAMESPACE::internal::string_length(FILENAME));
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(cpp::string_view(buf, len), cpp::string_view(LINK_VAL));

  ASSERT_THAT(LIBC_NAMESPACE::unlink(LINK), Succeeds(0));
}

TEST(LlvmLibcReadlinkatTest, ReadlinkInNonExistentPath) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  constexpr auto LEN = 8;
  char buf[LEN];
  ASSERT_THAT(
      LIBC_NAMESPACE::readlinkat(AT_FDCWD, "non-existent-link", buf, LEN),
      Fails(ENOENT));
}
