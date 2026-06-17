//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for realpath.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/limits_macros.h"
#include "src/__support/CPP/string.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/realpath.h"
#include "test/UnitTest/ErrnoCheckingTest.h"

namespace LIBC_NAMESPACE_DECL {

class LlvmLibcRealpathTest : public testing::ErrnoCheckingTest {
public:
  char *realpath_buffered(const char *path) { return realpath(path, buf_); }

  char *realpath_buffered(const cpp::string &path) {
    return realpath_buffered(path.c_str());
  }

  char buf_[PATH_MAX];
};

TEST_F(LlvmLibcRealpathTest, ErrorsWithInvalidArgIfNullPath) {
  ASSERT_EQ(realpath_buffered(nullptr), nullptr);
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST_F(LlvmLibcRealpathTest, ErrorsWithNoEntryIfEmptyPath) {
  ASSERT_EQ(realpath_buffered(""), nullptr);
  ASSERT_ERRNO_EQ(ENOENT);
}

TEST_F(LlvmLibcRealpathTest, OkIfPathIsExactlyMaxSize) {
  // PATH_MAX counts null terminator, so construct a path of size PATH_MAX-1.
  cpp::string s(PATH_MAX - 1, 'a');
  for (size_t i = 0; i < s.size(); i += NAME_MAX + 1)
    s[i] = '/';

  ASSERT_STREQ(realpath_buffered(s), s.c_str());
}

TEST_F(LlvmLibcRealpathTest, ErrorsWithNameTooLongIfPathExceedsMaxSize) {
  // PATH_MAX counts null terminator, so construct a path of size PATH_MAX.
  cpp::string s(PATH_MAX, 'a');
  for (size_t i = 0; i < s.size(); i += NAME_MAX + 1)
    s[i] = '/';

  ASSERT_EQ(realpath_buffered(s), nullptr);
  ASSERT_ERRNO_EQ(ENAMETOOLONG);
}

TEST_F(LlvmLibcRealpathTest, OkWhenComponentLengthIsExactlyNameMax) {
  cpp::string s(NAME_MAX + 1, 'a'); // +1 for leading "/"
  s[0] = '/';

  ASSERT_STREQ(realpath_buffered(s), s.c_str());
}

TEST_F(LlvmLibcRealpathTest, ErrorsIfComponentExceedsNameMax) {
  constexpr size_t COMPONENT_SIZE = NAME_MAX + 1;
  cpp::string s(COMPONENT_SIZE + 1, 'a'); // +1 for leading "/"
  s[0] = '/';

  ASSERT_EQ(realpath_buffered(s), nullptr);
  ASSERT_ERRNO_EQ(ENAMETOOLONG);
}

TEST_F(LlvmLibcRealpathTest, RootResolvesToRoot) {
  ASSERT_STREQ(realpath_buffered("/"), "/");
}

TEST_F(LlvmLibcRealpathTest, RootDotDotTraversalStaysAtRoot) {
  ASSERT_STREQ(realpath_buffered("/.."), "/");
}

TEST_F(LlvmLibcRealpathTest, SimpleAbsolutePath) {
  ASSERT_STREQ(realpath_buffered("/a/b/c"), "/a/b/c");
}

TEST_F(LlvmLibcRealpathTest, DotDotTraversesParent) {
  ASSERT_STREQ(realpath_buffered("/a/b/.."), "/a");
}

TEST_F(LlvmLibcRealpathTest, DotTraversalIsNop) {
  ASSERT_STREQ(realpath_buffered("/a/b/./"), "/a/b");
}

TEST_F(LlvmLibcRealpathTest, ConsecutiveSeparatorsIgnored) {
  ASSERT_STREQ(realpath_buffered("////a///..///a//"), "/a");
}

TEST_F(LlvmLibcRealpathTest, AllocatesResultWhenBufferIsNull) {
  char *result = LIBC_NAMESPACE::realpath("/a", nullptr);
  ASSERT_STREQ(result, "/a");
  ::free(result);
}

} // namespace LIBC_NAMESPACE_DECL
