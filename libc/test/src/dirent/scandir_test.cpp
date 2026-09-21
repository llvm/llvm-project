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
#include "src/stdio/fclose.h"
#include "src/stdio/fopen.h"
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
  if (LIBC_NAMESPACE::asprintf(&path, "%s%c%s", dir,
                               LIBC_NAMESPACE::path::SEPARATOR,
                               filename) == -1) {
    return nullptr;
  }
  return path;
}

bool create_empty_file(char *path) {
  FILE *file = LIBC_NAMESPACE::fopen(path, "w");
  if (file == nullptr) {
    return false;
  }

  if (LIBC_NAMESPACE::fclose(file) == -1) {
    return false;
  }
  return true;
}

char *create_temp_dir() {
  char *tmpl = LIBC_NAMESPACE::strdup(libc_make_test_file_path(TEMPLATE));
  if (tmpl == nullptr) {
    return nullptr;
  }
  return LIBC_NAMESPACE::mkdtemp(tmpl);
}

bool remove_temp_dir(char *dirpath) {
  if (LIBC_NAMESPACE::rmdir(dirpath) == -1) {
    return false;
  }
  free(dirpath);
  return true;
}

int alphasort(const struct dirent **a, const struct dirent **b) {
  return LIBC_NAMESPACE::strcoll((*a)->d_name, (*b)->d_name);
}

int omegasort(const struct dirent **a, const struct dirent **b) {
  return -LIBC_NAMESPACE::strcoll((*a)->d_name, (*b)->d_name);
}

int skip_hidden(const struct dirent *entry) { return entry->d_name[0] != '.'; }

void free_namelist(struct dirent **namelist, int size) {
  if (namelist == nullptr) {
    return;
  }

  for (int i = 0; i < size; ++i) {
    ::free(namelist[i]);
  }
  ::free(namelist);
}

TEST_F(LlvmLibcScandirTest, TestEmptyDir) {
  char *dirpath = create_temp_dir();
  ASSERT_NE(dirpath, nullptr);

  struct dirent **namelist;
  ASSERT_THAT(LIBC_NAMESPACE::scandir(dirpath, &namelist, nullptr, nullptr),
              Succeeds(ENTRIES_MIN));
  // Order of namelist is not guaranteed so we can't easily use ASSERT_STREQ
  ASSERT_TRUE((LIBC_NAMESPACE::strncmp(namelist[0]->d_name, ".", 1) == 0 &&
               LIBC_NAMESPACE::strncmp(namelist[1]->d_name, "..", 2) == 0) ||
              (LIBC_NAMESPACE::strncmp(namelist[0]->d_name, "..", 2) == 0 &&
               LIBC_NAMESPACE::strncmp(namelist[1]->d_name, ".", 1) == 0));

  // We also test that both orderings can't be true at the same time.
  ASSERT_FALSE((LIBC_NAMESPACE::strncmp(namelist[0]->d_name, ".", 1) == 0 &&
                LIBC_NAMESPACE::strncmp(namelist[1]->d_name, "..", 2) == 0) &&
               (LIBC_NAMESPACE::strncmp(namelist[0]->d_name, "..", 2) == 0 &&
                LIBC_NAMESPACE::strncmp(namelist[1]->d_name, ".", 1) == 0));

  free_namelist(namelist, ENTRIES_MIN);
  ASSERT_TRUE(remove_temp_dir(dirpath));
}

TEST_F(LlvmLibcScandirTest, TestDirFilter) {
  char *dirpath = create_temp_dir();
  ASSERT_NE(dirpath, nullptr);

  struct dirent **namelist;
  ASSERT_THAT(LIBC_NAMESPACE::scandir(dirpath, &namelist, skip_hidden, nullptr),
              Succeeds(0));

  free_namelist(namelist, 0);
  ASSERT_TRUE(remove_temp_dir(dirpath));
}

TEST_F(LlvmLibcScandirTest, TestDirSorted) {
  char *dirpath = create_temp_dir();
  ASSERT_NE(dirpath, nullptr);

  const char *files_to_create[] = {"d", "a", "1"};
  constexpr size_t NUM_FILES =
      sizeof(files_to_create) / sizeof(files_to_create[0]);
  char *filepaths[NUM_FILES];

  for (size_t i = 0; i < NUM_FILES; ++i) {
    filepaths[i] = join_path(dirpath, files_to_create[i]);
    ASSERT_NE(filepaths[i], nullptr);
    ASSERT_TRUE(create_empty_file(filepaths[i]));
  }

  struct dirent **namelist = nullptr;
  ASSERT_THAT(
      LIBC_NAMESPACE::scandir(dirpath, &namelist, skip_hidden, alphasort),
      Succeeds(3));

  ASSERT_STREQ(namelist[0]->d_name, "1");
  ASSERT_STREQ(namelist[1]->d_name, "a");
  ASSERT_STREQ(namelist[2]->d_name, "d");
  free_namelist(namelist, NUM_FILES);

  // Reverse alphanumeric sort in case the above sorting test passed on chance.
  namelist = nullptr;
  ASSERT_THAT(
      LIBC_NAMESPACE::scandir(dirpath, &namelist, skip_hidden, omegasort),
      Succeeds(3));

  ASSERT_STREQ(namelist[0]->d_name, "d");
  ASSERT_STREQ(namelist[1]->d_name, "a");
  ASSERT_STREQ(namelist[2]->d_name, "1");
  free_namelist(namelist, NUM_FILES);

  for (size_t i = 0; i < NUM_FILES; ++i) {
    ASSERT_THAT(LIBC_NAMESPACE::remove(filepaths[i]), Succeeds());
    ::free(filepaths[i]);
  }

  ASSERT_TRUE(remove_temp_dir(dirpath));
}

// While this test only checks for one type of ERROR, it really tests
// the error propagation from Dir::open. And as such we don't really
// have to test for every error inherited from Dir::open.
TEST_F(LlvmLibcScandirTest, TestBadDirname) {
  struct dirent **namelist;
  ASSERT_THAT(LIBC_NAMESPACE::scandir("", &namelist, NULL, NULL),
              Fails(ENOENT, -1));
}
