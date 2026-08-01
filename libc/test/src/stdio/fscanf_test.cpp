//===-- Unittests for fscanf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/stdio_macros.h"
#include "src/__support/CPP/string_view.h"
#include "src/stdio/fscanf.h"
#include "test/UnitTest/Test.h"

#ifndef LIBC_COPT_STDIO_USE_SYSTEM_FILE
#include "src/stdio/fclose.h"
#include "src/stdio/feof.h"
#include "src/stdio/ferror.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fwrite.h"
#endif // LIBC_COPT_STDIO_USE_SYSTEM_FILE

namespace scanf_test {
#ifndef LIBC_COPT_STDIO_USE_SYSTEM_FILE
using LIBC_NAMESPACE::fclose;
using LIBC_NAMESPACE::feof;
using LIBC_NAMESPACE::ferror;
using LIBC_NAMESPACE::fopen;
using LIBC_NAMESPACE::fwrite;
#else  // defined(LIBC_COPT_STDIO_USE_SYSTEM_FILE)
using ::fclose;
using ::feof;
using ::ferror;
using ::fopen;
using ::fwrite;
#endif // LIBC_COPT_STDIO_USE_SYSTEM_FILE
} // namespace scanf_test

TEST(LlvmLibcFScanfTest, WriteToFile) {
  const char *FILENAME = APPEND_LIBC_TEST("fscanf_output.test");
  auto FILE_PATH = libc_make_test_file_path(FILENAME);
  ::FILE *file = scanf_test::fopen(FILE_PATH, "w");
  ASSERT_FALSE(file == nullptr);

  int read;

  constexpr char simple[] = "A simple string with no conversions.\n";

  ASSERT_EQ(sizeof(simple) - 1,
            scanf_test::fwrite(simple, 1, sizeof(simple) - 1, file));

  constexpr char numbers[] = "1234567890\n";

  ASSERT_EQ(sizeof(numbers) - 1,
            scanf_test::fwrite(numbers, 1, sizeof(numbers) - 1, file));

  constexpr char numbers_and_more[] = "1234 and more\n";

  ASSERT_EQ(sizeof(numbers_and_more) - 1,
            scanf_test::fwrite(numbers_and_more, 1,
                               sizeof(numbers_and_more) - 1, file));

  read = LIBC_NAMESPACE::fscanf(file,
                                "Reading from a write-only file should fail.");
  EXPECT_NE(scanf_test::ferror(file), 0);
  EXPECT_EQ(read, EOF);

  ASSERT_EQ(0, scanf_test::fclose(file));

  file = scanf_test::fopen(FILE_PATH, "r");
  ASSERT_FALSE(file == nullptr);

  char data[50];
  read = LIBC_NAMESPACE::fscanf(file, "%[A-Za-z .\n]", data);
  ASSERT_EQ(read, 1);
  ASSERT_STREQ(simple, data);

  read = LIBC_NAMESPACE::fscanf(file, "%s", data);
  ASSERT_EQ(read, 1);
  ASSERT_EQ(LIBC_NAMESPACE::cpp::string_view(numbers, 10),
            LIBC_NAMESPACE::cpp::string_view(data));

  // The format string starts with a space to handle the fact that the %s leaves
  // a trailing \n and %c doesn't strip leading whitespace.
  read = LIBC_NAMESPACE::fscanf(file, " %50c", data);
  ASSERT_EQ(read, 1);
  ASSERT_EQ(
      LIBC_NAMESPACE::cpp::string_view(numbers_and_more),
      LIBC_NAMESPACE::cpp::string_view(data, sizeof(numbers_and_more) - 1));

  ASSERT_EQ(scanf_test::ferror(file), 0);
  ASSERT_EQ(scanf_test::fclose(file), 0);
}

TEST(LlvmLibcFScanfTest, ProcNetIfInet6Sample) {
  const char *FILENAME = APPEND_LIBC_TEST("proc_net_if_inet6.txt");
  auto FILE_PATH = libc_make_test_file_path(FILENAME);
  ::FILE *file = scanf_test::fopen(FILE_PATH, "w");
  ASSERT_FALSE(file == nullptr);

  // Sample contents of /proc/net/if_inet6 on Linux.
  constexpr char sample_contents[] =
      "fe80000000000000adaf264669baa4c7 02 40 20 80     ens4\n"
      "00000000000000000000000000000001 01 80 10 80       lo\n";
  constexpr char entry_format[] =
      "%4s%4s%4s%4s%4s%4s%4s%4s %08x %02x %02x %02x %20s\n";

  ASSERT_EQ(sizeof(sample_contents) - 1,
            scanf_test::fwrite(sample_contents, 1, sizeof(sample_contents) - 1,
                               file));
  ASSERT_EQ(0, scanf_test::fclose(file));

  file = scanf_test::fopen(FILE_PATH, "r");
  ASSERT_FALSE(file == nullptr);

  char addr6p[8][5];
  int if_idx, prefix, scope, dad_status;
  char devname[21];

  // Validate first entry.
  int ret = LIBC_NAMESPACE::fscanf(file, entry_format, addr6p[0], addr6p[1],
                                   addr6p[2], addr6p[3], addr6p[4], addr6p[5],
                                   addr6p[6], addr6p[7], &if_idx, &prefix,
                                   &scope, &dad_status, devname);
  ASSERT_EQ(ret, 13);
  EXPECT_STREQ(addr6p[0], "fe80");
  EXPECT_STREQ(addr6p[1], "0000");
  EXPECT_STREQ(addr6p[2], "0000");
  EXPECT_STREQ(addr6p[3], "0000");
  EXPECT_STREQ(addr6p[4], "adaf");
  EXPECT_STREQ(addr6p[5], "2646");
  EXPECT_STREQ(addr6p[6], "69ba");
  EXPECT_STREQ(addr6p[7], "a4c7");
  EXPECT_EQ(if_idx, 2);
  EXPECT_EQ(prefix, 64);
  EXPECT_EQ(scope, 32);
  EXPECT_EQ(dad_status, 128);
  EXPECT_STREQ(devname, "ens4");

  // Validate second entry.
  ret = LIBC_NAMESPACE::fscanf(file, entry_format, addr6p[0], addr6p[1],
                               addr6p[2], addr6p[3], addr6p[4], addr6p[5],
                               addr6p[6], addr6p[7], &if_idx, &prefix, &scope,
                               &dad_status, devname);
  ASSERT_EQ(ret, 13);
  for (int i = 0; i < 7; i++)
    EXPECT_STREQ(addr6p[i], "0000");
  EXPECT_STREQ(addr6p[7], "0001");
  EXPECT_EQ(if_idx, 1);
  EXPECT_EQ(prefix, 128);
  EXPECT_EQ(scope, 16);
  EXPECT_EQ(dad_status, 128);
  EXPECT_STREQ(devname, "lo");

  // No more entries, return EOF.
  ret = LIBC_NAMESPACE::fscanf(file, entry_format, addr6p[0], addr6p[1],
                               addr6p[2], addr6p[3], addr6p[4], addr6p[5],
                               addr6p[6], addr6p[7], &if_idx, &prefix, &scope,
                               &dad_status, devname);
  EXPECT_EQ(ret, EOF);
  EXPECT_NE(scanf_test::feof(file), 0);
  EXPECT_EQ(scanf_test::ferror(file), 0);

  ASSERT_EQ(scanf_test::fclose(file), 0);
}

TEST(LlvmLibcFScanfTest, EofPartialMatch) {
  const char *FILENAME = APPEND_LIBC_TEST("eof_partial_match.txt");
  auto FILE_PATH = libc_make_test_file_path(FILENAME);
  ::FILE *file = scanf_test::fopen(FILE_PATH, "w");
  ASSERT_FALSE(file == nullptr);

  constexpr char contents[] = "1 2 3";
  ASSERT_EQ(sizeof(contents) - 1,
            scanf_test::fwrite(contents, 1, sizeof(contents) - 1, file));
  ASSERT_EQ(0, scanf_test::fclose(file));

  file = scanf_test::fopen(FILE_PATH, "r");
  ASSERT_FALSE(file == nullptr);

  int vals[4] = {0};
  int ret = LIBC_NAMESPACE::fscanf(file, "%d %d %d %d", &vals[0], &vals[1],
                                   &vals[2], &vals[3]);
  // Returns 3 for number of matches despite EOF.
  EXPECT_EQ(ret, 3);
  EXPECT_EQ(vals[0], 1);
  EXPECT_EQ(vals[1], 2);
  EXPECT_EQ(vals[2], 3);

  EXPECT_NE(scanf_test::feof(file), 0);
  EXPECT_EQ(scanf_test::ferror(file), 0);

  ASSERT_EQ(scanf_test::fclose(file), 0);
}

TEST(LlvmLibcFScanfTest, MatchingErrors) {
  const char *FILENAME = APPEND_LIBC_TEST("matching_error_partial_match.txt");
  auto FILE_PATH = libc_make_test_file_path(FILENAME);
  ::FILE *file = scanf_test::fopen(FILE_PATH, "w");
  ASSERT_FALSE(file == nullptr);

  constexpr char contents[] = "one is 1 two is 2";
  ASSERT_EQ(sizeof(contents) - 1,
            scanf_test::fwrite(contents, 1, sizeof(contents) - 1, file));
  ASSERT_EQ(0, scanf_test::fclose(file));

  file = scanf_test::fopen(FILE_PATH, "r");
  ASSERT_FALSE(file == nullptr);

  int vals[2] = {0};
  // Immediate matching error.
  int ret =
      LIBC_NAMESPACE::fscanf(file, "zzz is %d two is %d", &vals[0], &vals[1]);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(scanf_test::feof(file), 0);
  EXPECT_EQ(scanf_test::ferror(file), 0);

  // Only the first item is matched.
  ret = LIBC_NAMESPACE::fscanf(file, "one is %d zzz is %d", &vals[0], &vals[1]);
  EXPECT_EQ(ret, 1);
  EXPECT_EQ(vals[0], 1);
  EXPECT_EQ(scanf_test::feof(file), 0);
  EXPECT_EQ(scanf_test::ferror(file), 0);

  // Second item is matched before EOF.
  ret = LIBC_NAMESPACE::fscanf(file, "two is %d zzz is %d", &vals[0], &vals[1]);
  EXPECT_EQ(ret, 1);
  EXPECT_EQ(vals[0], 2);
  EXPECT_NE(scanf_test::feof(file), 0);
  EXPECT_EQ(scanf_test::ferror(file), 0);

  ASSERT_EQ(scanf_test::fclose(file), 0);
}
