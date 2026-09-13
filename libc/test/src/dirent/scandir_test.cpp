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
#include "src/stdio/remove.h"
#include "src/stdlib/mkdtemp.h"
#include "src/string/strcoll.h"
#include "src/string/strdup.h"
#include "src/string/strncmp.h"
#include "src/unistd/rmdir.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcScandirTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

constexpr char TEMPLATE[] = "tmp_XXXXXX";

// A dir alwasys has '.' and '..' in it.
constexpr int ENTRIES_MIN = 2;

char *join_path(char *dir, const char *filename) {
  char *path = nullptr;
  if (LIBC_NAMESPACE::asprintf(&path, "%s%c%s", dir, LIBC_NAMESPACE::path::SEPARATOR, filename) == -1) {
    return nullptr;
  }
  return path;
}
bool create_empty_file(char *path) { FILE *file = LIBC_NAMESPACE::fopen(path, "w");
  if (file == nullptr) {
    return false;
  }

  if (LIBC_NAMESPACE::fclose(file) == -1) {
    return false;
  }
  return true;
}


int alphasort(const struct dirent **a, const struct dirent **b) {
  return LIBC_NAMESPACE::strcoll((*a)->d_name, (*b)->d_name);
}

int skip_hidden(const struct dirent *entry) {
    return entry->d_name[0] != '.';
}

TEST_F(LlvmLibcScandirTest, TestEmptyDir) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path(TEMPLATE));
  ASSERT_NE(tmpl, nullptr);
  ASSERT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Succeeds(tmpl));

  struct dirent **namelist;
  ASSERT_THAT(LIBC_NAMESPACE::scandir(tmpl, &namelist, nullptr, nullptr), Succeeds(ENTRIES_MIN));
  // Order of namelist is not guaranteed so we can't easily use ASSERT_STREQ
  ASSERT_TRUE(
      (LIBC_NAMESPACE::strncmp(namelist[0]->d_name, ".",  1) == 0 &&
       LIBC_NAMESPACE::strncmp(namelist[1]->d_name, "..", 2) == 0) ||
      (LIBC_NAMESPACE::strncmp(namelist[0]->d_name, "..", 2) == 0 &&
       LIBC_NAMESPACE::strncmp(namelist[1]->d_name, ".",  1) == 0));

  // We also test that both orderings can't be true at the same time.
  ASSERT_FALSE(
      (LIBC_NAMESPACE::strncmp(namelist[0]->d_name, ".",  1) == 0 &&
       LIBC_NAMESPACE::strncmp(namelist[1]->d_name, "..", 2) == 0) &&
      (LIBC_NAMESPACE::strncmp(namelist[0]->d_name, "..", 2) == 0 &&
       LIBC_NAMESPACE::strncmp(namelist[1]->d_name, ".",  1) == 0));

  ASSERT_THAT(LIBC_NAMESPACE::rmdir(tmpl), Succeeds());
  free(tmpl);
}

TEST_F(LlvmLibcScandirTest, TestDirFilter) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path(TEMPLATE));
  ASSERT_NE(tmpl, nullptr);
  ASSERT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Succeeds(tmpl));

  struct dirent **namelist;
  ASSERT_THAT(LIBC_NAMESPACE::scandir(tmpl, &namelist, skip_hidden, nullptr), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::rmdir(tmpl), Succeeds());
  free(tmpl);
}

TEST_F(LlvmLibcScandirTest, TestDirSorted) {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path(TEMPLATE));
  ASSERT_NE(tmpl, nullptr);
  ASSERT_THAT(LIBC_NAMESPACE::mkdtemp(tmpl), Succeeds(tmpl));

  char *path_d = join_path(tmpl, "d");
  ASSERT_TRUE(path_d != nullptr);
  ASSERT_TRUE(create_empty_file(path_d));

  char *path_a = join_path(tmpl, "a");
  ASSERT_TRUE(path_a != nullptr);
  ASSERT_TRUE(create_empty_file(path_a));

  char *path_1 = join_path(tmpl, "1");
  ASSERT_TRUE(path_1 != nullptr);
  ASSERT_TRUE(create_empty_file(path_1));

  struct dirent **namelist;
  ASSERT_THAT(LIBC_NAMESPACE::scandir(tmpl, &namelist, skip_hidden, alphasort), Succeeds(3));

  ASSERT_STREQ(namelist[0]->d_name, "1");
  ASSERT_STREQ(namelist[1]->d_name, "a");
  ASSERT_STREQ(namelist[2]->d_name, "d");

  ASSERT_THAT(LIBC_NAMESPACE::remove(path_d), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::remove(path_a), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::remove(path_1), Succeeds());

  free(path_d);
  free(path_a);
  free(path_1);

  ASSERT_THAT(LIBC_NAMESPACE::rmdir(tmpl), Succeeds());
  free(tmpl);
}

TEST_F(LlvmLibcScandirTest, TestBadDirname) {
  struct dirent **namelist;
  ASSERT_THAT(LIBC_NAMESPACE::scandir("", &namelist, NULL, NULL), Fails(ENOENT, -1));
}
