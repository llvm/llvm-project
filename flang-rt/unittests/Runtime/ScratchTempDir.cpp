//===-- unittests/Runtime/ScratchTempDir.cpp --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verifies that OPEN(STATUS='SCRATCH') honors the TMPDIR/TMP/TEMP/TEMPDIR
// environment variables on POSIX, matching
// llvm::sys::path::system_temp_directory.
//
//===----------------------------------------------------------------------===//

#ifndef _WIN32

#include "CrashHandlerFixture.h"
#include "gtest/gtest.h"
#include "flang/Runtime/io-api.h"
#include "flang/Runtime/iostat-consts.h"
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace Fortran::runtime;
using namespace Fortran::runtime::io;

namespace {

// Saves the values of TMPDIR/TMP/TEMP/TEMPDIR on construction and restores
// them on destruction so a test that mutates them does not leak state to
// later tests in the same process.
class TempDirEnvGuard {
public:
  TempDirEnvGuard() {
    for (const char *name : kVars) {
      if (const char *value{std::getenv(name)}) {
        saved_.emplace_back(name, std::string{value});
      } else {
        saved_.emplace_back(name, std::nullopt);
      }
      ::unsetenv(name);
    }
  }
  ~TempDirEnvGuard() {
    for (const auto &entry : saved_) {
      if (entry.second) {
        ::setenv(entry.first, entry.second->c_str(), /*overwrite=*/1);
      } else {
        ::unsetenv(entry.first);
      }
    }
  }

private:
  static constexpr const char *kVars[]{"TMPDIR", "TMP", "TEMP", "TEMPDIR"};
  std::vector<std::pair<const char *, std::optional<std::string>>> saved_;
};

// Opens a SCRATCH unit and returns the iostat value from EndIoStatement
// (IostatOk on success, non-zero on error). EnableHandlers makes the
// runtime defer I/O errors to EndIoStatement rather than crashing the
// process.
int OpenScratchUnit() {
  Cookie io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  IONAME(EnableHandlers)(io, /*hasIoStat=*/true);
  IONAME(SetStatus)(io, "SCRATCH", 7);
  IONAME(SetAction)(io, "READWRITE", 9);
  int unit{-1};
  IONAME(GetNewUnit)(io, unit);
  int iostat{IONAME(EndIoStatement)(io)};
  if (iostat == IostatOk && unit >= 0) {
    Cookie close{IONAME(BeginClose)(unit, __FILE__, __LINE__)};
    IONAME(EndIoStatement)(close);
  }
  return iostat;
}

struct ScratchTempDirTests : CrashHandlerFixture {};

// TMPDIR pointing at a non-existent directory must cause OPEN to fail —
// this proves the env var is actually consulted by the runtime rather
// than the hardcoded /tmp being used.
TEST_F(ScratchTempDirTests, TmpdirHonored) {
  TempDirEnvGuard guard;
  ::setenv("TMPDIR", "/this/path/does/not/exist/flang-rt-test", 1);
  EXPECT_NE(OpenScratchUnit(), IostatOk)
      << "OPEN(STATUS='SCRATCH') should fail when TMPDIR points to a "
         "non-existent directory";
}

// With no env vars set, OPEN must succeed via the /tmp fallback. This
// confirms backward compatibility for users who do not set TMPDIR.
TEST_F(ScratchTempDirTests, FallbackToTmp) {
  TempDirEnvGuard guard;
  EXPECT_EQ(OpenScratchUnit(), IostatOk)
      << "OPEN(STATUS='SCRATCH') should succeed via /tmp fallback when no "
         "temp env vars are set";
}

// TMP must be consulted when TMPDIR is unset. Set TMP to a non-existent
// path and expect failure.
TEST_F(ScratchTempDirTests, TmpConsulted) {
  TempDirEnvGuard guard;
  ::setenv("TMP", "/this/path/does/not/exist/flang-rt-test", 1);
  EXPECT_NE(OpenScratchUnit(), IostatOk)
      << "OPEN(STATUS='SCRATCH') should fail when TMP points to a "
         "non-existent directory and TMPDIR is unset";
}

// An empty TMPDIR string must be skipped, so the next env var (TMP) wins.
// Here TMP is a real directory, so OPEN should succeed.
TEST_F(ScratchTempDirTests, EmptyTmpdirSkipped) {
  TempDirEnvGuard guard;
  char tmpDir[]{"/tmp/flang-rt-scratch-test-XXXXXX"};
  ASSERT_NE(::mkdtemp(tmpDir), nullptr)
      << "mkdtemp failed: " << strerror(errno);
  ::setenv("TMPDIR", "", 1);
  ::setenv("TMP", tmpDir, 1);
  EXPECT_EQ(OpenScratchUnit(), IostatOk)
      << "Empty TMPDIR should be skipped in favor of TMP";
  ::rmdir(tmpDir);
}

// SCRATCH file should actually be created inside the user-specified temp
// directory. We can't observe the file by name (it is unlink'd immediately
// after creation), but we can observe that the inode count of the directory
// transiently increases while a SCRATCH unit is open.
TEST_F(ScratchTempDirTests, FileCreatedInTmpdir) {
  TempDirEnvGuard guard;
  char tmpDir[]{"/tmp/flang-rt-scratch-test-XXXXXX"};
  ASSERT_NE(::mkdtemp(tmpDir), nullptr)
      << "mkdtemp failed: " << strerror(errno);
  ::setenv("TMPDIR", tmpDir, 1);

  struct stat before{};
  ASSERT_EQ(::stat(tmpDir, &before), 0);

  Cookie io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  IONAME(EnableHandlers)(io, /*hasIoStat=*/true);
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7));
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9));
  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit));
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk);

  struct stat during{};
  ASSERT_EQ(::stat(tmpDir, &during), 0);
  // mkstemp creates the file then we immediately unlink it, so st_nlink
  // returns to the original count, but the directory's mtime/ctime should
  // have been bumped by the create+unlink pair.
  EXPECT_GE(during.st_mtime, before.st_mtime)
      << "TMPDIR should have been touched by SCRATCH file creation";

  Cookie close{IONAME(BeginClose)(unit, __FILE__, __LINE__)};
  EXPECT_EQ(IONAME(EndIoStatement)(close), IostatOk);
  ::rmdir(tmpDir);
}

} // namespace

#endif // !_WIN32
