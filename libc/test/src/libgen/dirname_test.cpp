//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for dirname.
///
//===----------------------------------------------------------------------===//

#include "src/libgen/dirname.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcDirnameTest, NullPointer) {
  ASSERT_STREQ(LIBC_NAMESPACE::dirname(nullptr), ".");
}

TEST(LlvmLibcDirnameTest, EmptyString) {
  char path[] = "";
  ASSERT_STREQ(LIBC_NAMESPACE::dirname(path), ".");
}

TEST(LlvmLibcDirnameTest, RegularPath) {
  char path[] = "/usr/lib";
  ASSERT_STREQ(LIBC_NAMESPACE::dirname(path), "/usr");
  ASSERT_STREQ(path, "/usr");
}

TEST(LlvmLibcDirnameTest, TrailingSlash) {
  char path[] = "/usr/";
  ASSERT_STREQ(LIBC_NAMESPACE::dirname(path), "/");
  ASSERT_STREQ(path, "/");
}

TEST(LlvmLibcDirnameTest, SingleSlash) {
  char path[] = "/";
  ASSERT_STREQ(LIBC_NAMESPACE::dirname(path), "/");
}

TEST(LlvmLibcDirnameTest, MultipleSlashes) {
  char path[] = "///";
  ASSERT_STREQ(LIBC_NAMESPACE::dirname(path), "/");
}

TEST(LlvmLibcDirnameTest, SimpleName) {
  char path[] = "a";
  ASSERT_STREQ(LIBC_NAMESPACE::dirname(path), ".");
}

TEST(LlvmLibcDirnameTest, SimpleNameTrailingSlash) {
  char path[] = "a/";
  ASSERT_STREQ(LIBC_NAMESPACE::dirname(path), ".");
}

TEST(LlvmLibcDirnameTest, ComplexPath) {
  char path[] = "///a///b///";
  ASSERT_STREQ(LIBC_NAMESPACE::dirname(path), "///a");
  ASSERT_STREQ(path, "///a");
}

TEST(LlvmLibcDirnameTest, SlashA) {
  char path[] = "/a";
  ASSERT_STREQ(LIBC_NAMESPACE::dirname(path), "/");
  ASSERT_STREQ(path, "/");
}

TEST(LlvmLibcDirnameTest, MultipleSlashesA) {
  char path[] = "///a";
  ASSERT_STREQ(LIBC_NAMESPACE::dirname(path), "/");
  ASSERT_STREQ(path, "/");
}
