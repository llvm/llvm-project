//===- JSONFormatTest.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test fixture and helpers for SSAF JSON serialization format unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMATTEST_H
#define LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMATTEST_H

#include "clang/Analysis/Scalable/Serialization/JSONFormat.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>

using namespace clang::ssaf;
using namespace llvm;
using ::testing::AllOf;
using ::testing::HasSubstr;

// ============================================================================
// Test Fixture
// ============================================================================

class JSONFormatTest : public ::testing::Test {
public:
  using PathString = SmallString<128>;

protected:
  SmallString<128> TestDir;

  void SetUp() override {
    std::error_code EC =
        sys::fs::createUniqueDirectory("json-format-test", TestDir);
    ASSERT_FALSE(EC) << "Failed to create temp directory: " << EC.message();
  }

  void TearDown() override { sys::fs::remove_directories(TestDir); }

  PathString makePath(StringRef FileOrDirectoryName) const {
    PathString FullPath = TestDir;
    sys::path::append(FullPath, FileOrDirectoryName);

    return FullPath;
  }

  PathString makePath(StringRef Dir, StringRef FileName) const {
    PathString FullPath = TestDir;
    sys::path::append(FullPath, Dir, FileName);

    return FullPath;
  }

  Expected<PathString> makeDirectory(StringRef DirectoryName) const {
    PathString DirPath = makePath(DirectoryName);

    std::error_code EC = sys::fs::create_directory(DirPath);
    if (EC) {
      return createStringError(EC, "Failed to create directory '%s': %s",
                               DirPath.c_str(), EC.message().c_str());
    }

    return DirPath;
  }

  Expected<PathString> makeSymlink(StringRef TargetFileName,
                                   StringRef SymlinkFileName) const {
    PathString TargetPath = makePath(TargetFileName);
    PathString SymlinkPath = makePath(SymlinkFileName);

    std::error_code EC = sys::fs::create_link(TargetPath, SymlinkPath);
    if (EC) {
      return createStringError(EC, "Failed to create symlink '%s' -> '%s': %s",
                               SymlinkPath.c_str(), TargetPath.c_str(),
                               EC.message().c_str());
    }

    return SymlinkPath;
  }

  llvm::Error setPermission(StringRef FileName,
                            const sys::fs::perms Perms) const {
    PathString Path = makePath(FileName);

    std::error_code EC = sys::fs::setPermissions(Path, Perms);
    if (EC) {
      return createStringError(EC, "Failed to set permissions on '%s': %s",
                               Path.c_str(), EC.message().c_str());
    }

    return llvm::Error::success();
  }

  Expected<json::Value> readJSONFromFile(StringRef FileName) const {
    PathString FilePath = makePath(FileName);

    auto BufferOrError = MemoryBuffer::getFile(FilePath);
    if (!BufferOrError) {
      return createStringError(BufferOrError.getError(),
                               "Failed to read file: %s", FilePath.c_str());
    }

    Expected<json::Value> ExpectedValue =
        json::parse(BufferOrError.get()->getBuffer());
    if (!ExpectedValue)
      return ExpectedValue.takeError();

    return *ExpectedValue;
  }

  Expected<PathString> writeJSON(StringRef JSON, StringRef FileName) const {
    PathString FilePath = makePath(FileName);

    std::error_code EC;
    llvm::raw_fd_ostream OS(FilePath, EC);
    if (EC) {
      return createStringError(EC, "Failed to create file '%s': %s",
                               FilePath.c_str(), EC.message().c_str());
    }

    OS << JSON;
    OS.close();

    if (OS.has_error()) {
      return createStringError(OS.error(), "Failed to write to file '%s': %s",
                               FilePath.c_str(), OS.error().message().c_str());
    }

    return FilePath;
  }
};

#endif // LLVM_CLANG_UNITTESTS_ANALYSIS_SCALABLE_SERIALIZATION_JSONFORMATTEST_H
