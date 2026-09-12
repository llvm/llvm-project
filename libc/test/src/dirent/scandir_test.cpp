//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for scandir.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_dirent.h"
#include "src/__support/OSUtil/path.h"
#include "src/dirent/scandir.h"
#include "src/stdio/asprintf.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fclose.h"
#include "src/stdlib/mkdtemp.h"
#include "src/string/strdup.h"
#include "src/unistd/rmdir.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcScandirTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

constexpr char TEMPLATE[] = "tmp_XXXXXX";

// A dir alwasys has '.' and '..' in it.
constexpr int MINIMUM_ENTRIES = 2;

bool create_file(char *dir, const char *name) {
  char *path = nullptr;

  if (LIBC_NAMESPACE::asprintf(&path, "%s%c%s", dir, LIBC_NAMESPACE::path::SEPARATOR, name) == -1) {
    return false;
  }

  FILE *file = LIBC_NAMESPACE::fopen(path, "w");
  if (file == nullptr) {
    return false;
  }

  if (LIBC_NAMESPACE::fclose(file) == -1) {
    return false;
  }

  return true;
}


TEST_F(LlvmLibcScandirTest, TestBasic) {

  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path(TEMPLATE));
  ASSERT_NE(tmpl, nullptr);
  ASSERT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Succeeds(tmpl));

  const char *filename_a = libc_make_test_file_path("a");
  const char *filename_b = libc_make_test_file_path("file_b");

  ASSERT_TRUE(create_file(tmpl, filename_a));
  ASSERT_TRUE(create_file(tmpl, filename_b));

  struct dirent **namelist;
  ASSERT_THAT(LIBC_NAMESPACE::scandir(tmpl, &namelist, NULL, NULL), Succeeds(MINIMUM_ENTRIES + 2));

  // TODO: Implement file deletion!
  /*
  ASSERT_THAT(LIBC_NAMESPACE::rmdir(tmpl), Succeeds());
  free(tmpl);
  */
}


TEST_F(LlvmLibcScandirTest, TestBadDirname) {
  struct dirent **namelist;
  ASSERT_THAT(LIBC_NAMESPACE::scandir("", &namelist, NULL, NULL), Fails(ENOENT, -1));
}
