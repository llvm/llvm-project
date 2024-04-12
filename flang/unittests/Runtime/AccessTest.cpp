//===-- flang/unittests/Runtime/AccessTest.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: ACCESS is not yet implemented on Windows
#ifndef _WIN32

#include "CrashHandlerFixture.h"
#include "gtest/gtest.h"
#include "flang/Runtime/extensions.h"

#include <fcntl.h>
#include <filesystem>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

struct AccessTests : public CrashHandlerFixture {};

struct AccessType {
  bool read{false};
  bool write{false};
  bool execute{false};
  bool exists{false};
};

} // namespace

static std::string addPIDSuffix(const char *name) {
  std::stringstream ss;
  ss << name;
  ss << '.';

  ss << getpid();

  return ss.str();
}

static std::filesystem::path createTemporaryFile(
    const char *name, const AccessType &accessType) {
  std::filesystem::path path{
      std::filesystem::temp_directory_path() / addPIDSuffix(name)};

  // O_CREAT | O_EXCL enforces that this file is newly created by this call.
  // This feels risky. If we don't have permission to create files in the
  // temporary directory or if the files already exist, the test will fail.
  // But we can't use std::tmpfile() because we need a path to the file and
  // to control the filesystem permissions
  mode_t mode{0};
  if (accessType.read) {
    mode |= S_IRUSR;
  }
  if (accessType.write) {
    mode |= S_IWUSR;
  }
  if (accessType.execute) {
    mode |= S_IXUSR;
  }

  int file = open(path.c_str(), O_CREAT | O_EXCL, mode);
  if (file == -1) {
    return {};
  }

  close(file);

  return path;
}

static std::int64_t callAccess(
    const std::filesystem::path &path, const AccessType &accessType) {
  const char *cpath{path.c_str()};
  std::int64_t pathlen = std::strlen(cpath);

  std::string mode;
  if (accessType.read) {
    mode += 'r';
  }
  if (accessType.write) {
    mode += 'w';
  }
  if (accessType.execute) {
    mode += 'x';
  }
  if (accessType.exists) {
    mode += ' ';
  }

  const char *cmode = mode.c_str();
  std::int64_t modelen = std::strlen(cmode);

  return FORTRAN_PROCEDURE_NAME(access)(cpath, pathlen, cmode, modelen);
}

TEST(AccessTests, TestExists) {
  AccessType accessType;
  accessType.exists = true;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_EQ(res, 0);
}

TEST(AccessTests, TestNotExists) {
  std::filesystem::path nonExistant{addPIDSuffix(__func__)};
  ASSERT_FALSE(std::filesystem::exists(nonExistant));

  AccessType accessType;
  accessType.exists = true;
  std::int64_t res = callAccess(nonExistant, accessType);

  ASSERT_NE(res, 0);
}

TEST(AccessTests, TestRead) {
  AccessType accessType;
  accessType.read = true;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_EQ(res, 0);
}

TEST(AccessTests, TestNotRead) {
  AccessType accessType;
  accessType.read = false;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  accessType.read = true;
  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_NE(res, 0);
}

TEST(AccessTests, TestWrite) {
  AccessType accessType;
  accessType.write = true;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_EQ(res, 0);
}

TEST(AccessTests, TestNotWrite) {
  AccessType accessType;
  accessType.write = false;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  accessType.write = true;
  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_NE(res, 0);
}

TEST(AccessTests, TestReadWrite) {
  AccessType accessType;
  accessType.read = true;
  accessType.write = true;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_EQ(res, 0);
}

TEST(AccessTests, TestNotReadWrite0) {
  AccessType accessType;
  accessType.read = false;
  accessType.write = false;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  accessType.read = true;
  accessType.write = true;
  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_NE(res, 0);
}

TEST(AccessTests, TestNotReadWrite1) {
  AccessType accessType;
  accessType.read = true;
  accessType.write = false;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  accessType.read = true;
  accessType.write = true;
  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_NE(res, 0);
}

TEST(AccessTests, TestNotReadWrite2) {
  AccessType accessType;
  accessType.read = false;
  accessType.write = true;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  accessType.read = true;
  accessType.write = true;
  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_NE(res, 0);
}

TEST(AccessTests, TestExecute) {
  AccessType accessType;
  accessType.execute = true;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_EQ(res, 0);
}

TEST(AccessTests, TestNotExecute) {
  AccessType accessType;
  accessType.execute = false;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  accessType.execute = true;
  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_NE(res, 0);
}

TEST(AccessTests, TestRWX) {
  AccessType accessType;
  accessType.read = true;
  accessType.write = true;
  accessType.execute = true;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_EQ(res, 0);
}

TEST(AccessTests, TestNotRWX0) {
  AccessType accessType;
  accessType.read = false;
  accessType.write = false;
  accessType.execute = false;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  accessType.read = true;
  accessType.write = true;
  accessType.execute = true;
  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_NE(res, 0);
}

TEST(AccessTests, TestNotRWX1) {
  AccessType accessType;
  accessType.read = true;
  accessType.write = false;
  accessType.execute = false;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  accessType.read = true;
  accessType.write = true;
  accessType.execute = true;
  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_NE(res, 0);
}

TEST(AccessTests, TestNotRWX2) {
  AccessType accessType;
  accessType.read = true;
  accessType.write = true;
  accessType.execute = false;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  accessType.read = true;
  accessType.write = true;
  accessType.execute = true;
  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_NE(res, 0);
}

TEST(AccessTests, TestNotRWX3) {
  AccessType accessType;
  accessType.read = true;
  accessType.write = false;
  accessType.execute = true;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  accessType.read = true;
  accessType.write = true;
  accessType.execute = true;
  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_NE(res, 0);
}

TEST(AccessTests, TestNotRWX4) {
  AccessType accessType;
  accessType.read = false;
  accessType.write = true;
  accessType.execute = true;

  std::filesystem::path path = createTemporaryFile(__func__, accessType);
  ASSERT_FALSE(path.empty());

  accessType.read = true;
  accessType.write = true;
  accessType.execute = true;
  std::int64_t res = callAccess(path, accessType);

  std::filesystem::remove(path);

  ASSERT_NE(res, 0);
}

#endif // !_WIN32
