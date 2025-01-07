//===-- Unittests for getopt ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/unistd/getopt.h"
#include "test/UnitTest/Test.h"

#include "src/__support/CPP/array.h"
#include "src/stdio/fflush.h"
#include "src/stdio/fopencookie.h"

using LIBC_NAMESPACE::cpp::array;

namespace test_globals {
char *optarg;
int optind = 1;
int optopt;
int opterr = 1;

unsigned optpos;
} // namespace test_globals

// This can't be a constructor because it will get run before the constructor
// which sets the default state in getopt.
void set_state(FILE *errstream) {
  LIBC_NAMESPACE::impl::set_getopt_state(
      &test_globals::optarg, &test_globals::optind, &test_globals::optopt,
      &test_globals::optpos, &test_globals::opterr, errstream);
}

static void my_memcpy(char *dest, const char *src, size_t size) {
  for (size_t i = 0; i < size; i++)
    dest[i] = src[i];
}

ssize_t cookie_write(void *cookie, const char *buf, size_t size) {
  char **pos = static_cast<char **>(cookie);
  my_memcpy(*pos, buf, size);
  *pos += size;
  return size;
}

static cookie_io_functions_t cookie{nullptr, &cookie_write, nullptr, nullptr};

// TODO: <stdio> could be either llvm-libc's or the system libc's. The former
// doesn't currently support fmemopen but does have fopencookie. In the future
// just use that instead. This memopen does no error checking for the size
// of the buffer, etc.
FILE *memopen(char **pos) {
  return LIBC_NAMESPACE::fopencookie(pos, "w", cookie);
}

struct LlvmLibcGetoptTest : public LIBC_NAMESPACE::testing::Test {
  FILE *errstream;
  char buf[256];
  char *pos = buf;

  void reset_errstream() { pos = buf; }
  const char *get_error_msg() {
    LIBC_NAMESPACE::fflush(errstream);
    return buf;
  }

  void SetUp() override {
    ASSERT_TRUE(!!(errstream = memopen(&pos)));
    set_state(errstream);
    ASSERT_EQ(test_globals::optind, 1);
  }

  void TearDown() override {
    test_globals::optind = 1;
    test_globals::opterr = 1;
  }
};

// This is safe because getopt doesn't currently permute argv like GNU's getopt
// does so this just helps silence warnings.
char *operator"" _c(const char *c, size_t) { return const_cast<char *>(c); }

TEST_F(LlvmLibcGetoptTest, NoMatch) {
  array<char *, 3> argv{"prog"_c, "arg1"_c, nullptr};

  // optind >= argc
  EXPECT_EQ(LIBC_NAMESPACE::getopt(1, argv.data(), "..."), -1);

  // argv[optind] == nullptr
  test_globals::optind = 2;
  EXPECT_EQ(LIBC_NAMESPACE::getopt(100, argv.data(), "..."), -1);

  // argv[optind][0] != '-'
  test_globals::optind = 1;
  EXPECT_EQ(LIBC_NAMESPACE::getopt(2, argv.data(), "a"), -1);
  ASSERT_EQ(test_globals::optind, 1);

  // argv[optind] == "-"
  argv[1] = "-"_c;
  EXPECT_EQ(LIBC_NAMESPACE::getopt(2, argv.data(), "a"), -1);
  ASSERT_EQ(test_globals::optind, 1);

  // argv[optind] == "--", then return -1 and incremement optind
  argv[1] = "--"_c;
  EXPECT_EQ(LIBC_NAMESPACE::getopt(2, argv.data(), "a"), -1);
  EXPECT_EQ(test_globals::optind, 2);
}

TEST_F(LlvmLibcGetoptTest, WrongMatch) {
  array<char *, 3> argv{"prog"_c, "-b"_c, nullptr};

  EXPECT_EQ(LIBC_NAMESPACE::getopt(2, argv.data(), "a"), int('?'));
  EXPECT_EQ(test_globals::optopt, (int)'b');
  EXPECT_EQ(test_globals::optind, 1);
  EXPECT_STREQ(get_error_msg(), "prog: illegal option -- b\n");
}

TEST_F(LlvmLibcGetoptTest, OpterrFalse) {
  array<char *, 3> argv{"prog"_c, "-b"_c, nullptr};

  test_globals::opterr = 0;
  set_state(errstream);
  EXPECT_EQ(LIBC_NAMESPACE::getopt(2, argv.data(), "a"), int('?'));
  EXPECT_EQ(test_globals::optopt, (int)'b');
  EXPECT_EQ(test_globals::optind, 1);
  EXPECT_STREQ(get_error_msg(), "");
}

TEST_F(LlvmLibcGetoptTest, MissingArg) {
  array<char *, 3> argv{"prog"_c, "-b"_c, nullptr};

  EXPECT_EQ(LIBC_NAMESPACE::getopt(2, argv.data(), ":b:"), (int)':');
  ASSERT_EQ(test_globals::optind, 1);
  EXPECT_STREQ(get_error_msg(), "prog: option requires an argument -- b\n");
  reset_errstream();
  EXPECT_EQ(LIBC_NAMESPACE::getopt(2, argv.data(), "b:"), int('?'));
  EXPECT_EQ(test_globals::optind, 1);
  EXPECT_STREQ(get_error_msg(), "prog: option requires an argument -- b\n");
}

TEST_F(LlvmLibcGetoptTest, ParseArgInCurrent) {
  array<char *, 3> argv{"prog"_c, "-barg"_c, nullptr};

  EXPECT_EQ(LIBC_NAMESPACE::getopt(2, argv.data(), "b:"), (int)'b');
  EXPECT_STREQ(test_globals::optarg, "arg");
  EXPECT_EQ(test_globals::optind, 2);
}

TEST_F(LlvmLibcGetoptTest, ParseArgInNext) {
  array<char *, 4> argv{"prog"_c, "-b"_c, "arg"_c, nullptr};

  EXPECT_EQ(LIBC_NAMESPACE::getopt(3, argv.data(), "b:"), (int)'b');
  EXPECT_STREQ(test_globals::optarg, "arg");
  EXPECT_EQ(test_globals::optind, 3);
}

TEST_F(LlvmLibcGetoptTest, ParseMutliInOne) {
  array<char *, 3> argv{"prog"_c, "-abc"_c, nullptr};

  EXPECT_EQ(LIBC_NAMESPACE::getopt(2, argv.data(), "abc"), (int)'a');
  ASSERT_EQ(test_globals::optind, 1);
  EXPECT_EQ(LIBC_NAMESPACE::getopt(2, argv.data(), "abc"), (int)'b');
  ASSERT_EQ(test_globals::optind, 1);
  EXPECT_EQ(LIBC_NAMESPACE::getopt(2, argv.data(), "abc"), (int)'c');
  EXPECT_EQ(test_globals::optind, 2);
}
