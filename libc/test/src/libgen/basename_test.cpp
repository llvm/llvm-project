//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for basename.
///
//===----------------------------------------------------------------------===//

#include "src/libgen/basename.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcBasenameTest, NullPointer) {
  ASSERT_STREQ(LIBC_NAMESPACE::basename(nullptr), ".");
}

TEST(LlvmLibcBasenameTest, EmptyString) {
  char path[] = "";
  ASSERT_STREQ(LIBC_NAMESPACE::basename(path), ".");
}

TEST(LlvmLibcBasenameTest, RegularPath) {
  char path[] = "/usr/lib";
  ASSERT_STREQ(LIBC_NAMESPACE::basename(path), "lib");
}

TEST(LlvmLibcBasenameTest, TrailingSlash) {
  char path[] = "/usr/";
  ASSERT_STREQ(LIBC_NAMESPACE::basename(path), "usr");
  ASSERT_STREQ(path, "/usr");
}

TEST(LlvmLibcBasenameTest, SingleSlash) {
  char path[] = "/";
  ASSERT_STREQ(LIBC_NAMESPACE::basename(path), "/");
}

TEST(LlvmLibcBasenameTest, MultipleSlashes) {
  char path[] = "///";
  ASSERT_STREQ(LIBC_NAMESPACE::basename(path), "/");
}

TEST(LlvmLibcBasenameTest, SimpleName) {
  char path[] = "a";
  ASSERT_STREQ(LIBC_NAMESPACE::basename(path), "a");
}

TEST(LlvmLibcBasenameTest, SimpleNameTrailingSlash) {
  char path[] = "a/";
  ASSERT_STREQ(LIBC_NAMESPACE::basename(path), "a");
  ASSERT_STREQ(path, "a");
}

TEST(LlvmLibcBasenameTest, ComplexPath) {
  char path[] = "///a///";
  ASSERT_STREQ(LIBC_NAMESPACE::basename(path), "a");
  ASSERT_STREQ(path, "///a");
}
