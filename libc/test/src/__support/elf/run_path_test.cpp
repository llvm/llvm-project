//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for run path expansion
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/__support/elf/run_path.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::elf::directory_length;
using LIBC_NAMESPACE::elf::executable_origin;
using LIBC_NAMESPACE::elf::expand_run_path;

namespace {

// Expands a whole entry and reports what came out, or an empty view if it
// could not be expanded.
string_view expand(const char *entry, const char *origin, char *buffer,
                   size_t capacity) {
  size_t length = 0;
  while (entry[length] != '\0')
    ++length;
  if (!expand_run_path(entry, length, origin, buffer, capacity))
    return string_view();
  return string_view(buffer);
}

} // anonymous namespace

TEST(LlvmLibcElfRunPathTest, EntryWithoutOriginIsCopied) {
  char buffer[64];
  EXPECT_TRUE(expand("/usr/lib", nullptr, buffer, sizeof(buffer)) ==
              string_view("/usr/lib"));
}

TEST(LlvmLibcElfRunPathTest, OriginBecomesTheObjectsDirectory) {
  char buffer[64];
  EXPECT_TRUE(expand("$ORIGIN/../lib", "/opt/app/bin/tool", buffer,
                     sizeof(buffer)) == string_view("/opt/app/bin/../lib"));
  EXPECT_TRUE(expand("${ORIGIN}/lib", "/opt/app/bin/tool", buffer,
                     sizeof(buffer)) == string_view("/opt/app/bin/lib"));
}

TEST(LlvmLibcElfRunPathTest, OriginOnItsOwn) {
  char buffer[64];
  EXPECT_TRUE(expand("$ORIGIN", "/opt/app/tool", buffer, sizeof(buffer)) ==
              string_view("/opt/app"));
}

TEST(LlvmLibcElfRunPathTest, OriginAppearsMoreThanOnce) {
  char buffer[64];
  EXPECT_TRUE(expand("$ORIGIN/a:$ORIGIN/b", "/x/y/z", buffer, sizeof(buffer)) ==
              string_view("/x/y/a:/x/y/b"));
}

TEST(LlvmLibcElfRunPathTest, ObjectInTheWorkingDirectory) {
  char buffer[64];
  // Nothing names a directory, so $ORIGIN is the directory the name was
  // resolved against.
  EXPECT_TRUE(expand("$ORIGIN/lib", "tool", buffer, sizeof(buffer)) ==
              string_view("./lib"));
}

TEST(LlvmLibcElfRunPathTest, ObjectAtTheRoot) {
  char buffer[64];
  EXPECT_TRUE(expand("$ORIGIN/lib", "/tool", buffer, sizeof(buffer)) ==
              string_view("//lib"));
}

TEST(LlvmLibcElfRunPathTest, DollarThatIsNotOrigin) {
  char buffer[64];
  EXPECT_TRUE(expand("/opt/$LIB", "/x/y", buffer, sizeof(buffer)) ==
              string_view("/opt/$LIB"));
  EXPECT_TRUE(expand("$ORIGINAL", "/x/y", buffer, sizeof(buffer)) ==
              string_view("/xAL"));
}

TEST(LlvmLibcElfRunPathTest, RefusesWhatItCannotExpand) {
  char buffer[64];
  // No origin to put in.
  EXPECT_TRUE(expand("$ORIGIN/lib", nullptr, buffer, sizeof(buffer)).empty());
  // Nothing to expand at all.
  EXPECT_TRUE(expand("", "/x/y", buffer, sizeof(buffer)).empty());
  // Does not fit.
  char small[8];
  EXPECT_TRUE(
      expand("$ORIGIN/lib", "/a/very/long/path/tool", small, sizeof(small))
          .empty());
}

TEST(LlvmLibcElfRunPathTest, DirectoryLength) {
  EXPECT_EQ(directory_length("/opt/app/tool"), size_t(8));
  EXPECT_EQ(directory_length("tool"), size_t(0));
  EXPECT_EQ(directory_length("/tool"), size_t(1));
  EXPECT_EQ(directory_length(""), size_t(0));
}

// The executable's origin is the file it really is, which /proc gives as an
// absolute path, rather than whatever it was started by.
TEST(LlvmLibcElfRunPathTest, ExecutableOriginIsTheFileItReallyIs) {
  char buffer[4096];
  const char *origin =
      executable_origin("/started/through/a/link", buffer, sizeof(buffer));
  ASSERT_TRUE(origin == buffer);
  EXPECT_EQ(origin[0], '/');
}

TEST(LlvmLibcElfRunPathTest, ExecutableOriginWithoutRoomIsHowItWasStarted) {
  char buffer[4];
  EXPECT_TRUE(
      string_view(executable_origin("/bin/tool", buffer, sizeof(buffer))) ==
      string_view("/bin/tool"));
}
