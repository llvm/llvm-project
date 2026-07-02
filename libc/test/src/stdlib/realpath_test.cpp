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
#include "hdr/types/size_t.h"
#include "src/__support/CPP/string.h"
#include "src/__support/OSUtil/path.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/realpath.h"
#include "test/UnitTest/ErrnoCheckingTest.h"

namespace cpp = LIBC_NAMESPACE::cpp;
namespace path = LIBC_NAMESPACE::path;
using LIBC_NAMESPACE::testing::ErrnoCheckingTest;

// This test assumes the following values, so fail early if they mismatch.
static_assert(path::SEPARATOR == '/');
static_assert(path::CURRENT_DIR_COMPONENT == ".");
static_assert(path::PARENT_DIR_COMPONENT == "..");

class LlvmLibcRealpathTest : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
public:
  char *realpath_buffered(const char *path) {
    return LIBC_NAMESPACE::realpath(path, buf_);
  }

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

TEST_F(LlvmLibcRealpathTest, OkIfPathArgIsExactlyMaxSize) {
  // PATH_MAX counts null terminator, so construct a path of size PATH_MAX-1.
  cpp::string s(PATH_MAX - 1, '/');
  for (size_t i = 1; i < s.size(); i += 2)
    s[i] = '.';

  ASSERT_STREQ(realpath_buffered(s), "/");
}

TEST_F(LlvmLibcRealpathTest, OkIfResolvedPathIsExactlyMaxSize) {
  // PATH_MAX counts null terminator, so construct a path of size PATH_MAX-1.
  cpp::string s(PATH_MAX - 1, 'a');
  for (size_t i = 0; i < s.size() - 1; i += 8)
    s[i] = '/';

  ASSERT_STREQ(realpath_buffered(s), s.c_str());
}

TEST_F(LlvmLibcRealpathTest, ErrorsWithNameTooLongIfPathArgExceedsMaxSize) {
  // PATH_MAX counts null terminator, so construct a path of size PATH_MAX.
  cpp::string s(PATH_MAX, '/');
  for (size_t i = 1; i < s.size(); i += 2)
    s[i] = '.';

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
