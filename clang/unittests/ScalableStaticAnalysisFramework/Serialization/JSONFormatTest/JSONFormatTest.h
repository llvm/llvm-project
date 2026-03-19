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

#ifndef LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSISFRAMEWORK_SERIALIZATION_JSONFORMATTEST_JSONFORMATTEST_H
#define LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSISFRAMEWORK_SERIALIZATION_JSONFORMATTEST_JSONFORMATTEST_H

#include "TestFixture.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <string>

namespace clang::ssaf {

// ============================================================================
// Test Fixture
// ============================================================================

class JSONFormatTest : public TestFixture {
public:
  using PathString = llvm::SmallString<128>;

protected:
  llvm::SmallString<128> TestDir;

  void SetUp() override;

  void TearDown() override;

  PathString makePath(llvm::StringRef FileOrDirectoryName) const;

  PathString makePath(llvm::StringRef Dir, llvm::StringRef FileName) const;

  llvm::Expected<PathString> makeDirectory(llvm::StringRef DirectoryName) const;

  llvm::Expected<PathString> makeSymlink(llvm::StringRef TargetFileName,
                                         llvm::StringRef SymlinkFileName) const;

  llvm::Error setPermission(llvm::StringRef FileName,
                            llvm::sys::fs::perms Perms) const;

  // Returns true if Unix file permission checks are enforced in this
  // environment. Returns false if running as root (uid 0), or if a probe
  // file with read permission removed can still be opened, indicating that
  // permission checks are not enforced (e.g. certain container setups).
  // Tests that rely on permission-based failure conditions should skip
  // themselves when this returns false.
  bool permissionsAreEnforced() const;

  llvm::Expected<llvm::json::Value>
  readJSONFromFile(llvm::StringRef FileName) const;

  llvm::Expected<PathString> writeJSON(llvm::StringRef JSON,
                                       llvm::StringRef FileName) const;
};

// ============================================================================
// SummaryOps - Parameterization struct for TUSummary/LUSummary test suites
// ============================================================================

struct SummaryOps {
  // Suffix appended to the test name by GTest to identify this parameter
  // instantiation (e.g. "Resolved", "Encoding").
  std::string GTestInstantiationSuffix;

  // Human-readable name of the summary class under test (e.g. "TUSummary").
  // Used in diagnostic messages to identify which summary type an error
  // originated from.
  std::string SummaryClassName;

  std::function<llvm::Error(llvm::StringRef FilePath)> ReadFromFile;

  std::function<llvm::Error(llvm::StringRef FilePath)> WriteEmpty;

  std::function<llvm::Error(llvm::StringRef InputFilePath,
                            llvm::StringRef OutputFilePath)>
      ReadWriteRoundTrip;
};

// ============================================================================
// SummaryTest Test Fixture
// ============================================================================

class SummaryTest : public JSONFormatTest,
                    public ::testing::WithParamInterface<SummaryOps> {
protected:
  llvm::Error readFromString(llvm::StringRef JSON,
                             llvm::StringRef FileName = "test.json") const;

  llvm::Error readFromFile(llvm::StringRef FileName) const;

  llvm::Error writeEmpty(llvm::StringRef FileName) const;

  llvm::Error readWriteRoundTrip(llvm::StringRef InputFileName,
                                 llvm::StringRef OutputFileName) const;

  void readWriteCompare(llvm::StringRef JSON) const;
};

// ============================================================================
// First Test Analysis - Simple analysis for testing JSON serialization.
// ============================================================================

struct PairsEntitySummaryForJSONFormatTest final : EntitySummary {

  SummaryName getSummaryName() const override {
    return SummaryName("PairsEntitySummaryForJSONFormatTest");
  }

  std::vector<std::pair<EntityId, EntityId>> Pairs;
};

// ============================================================================
// Second Test Analysis - Simple analysis for multi-summary round-trip tests.
// ============================================================================

struct TagsEntitySummaryForJSONFormatTest final : EntitySummary {
  SummaryName getSummaryName() const override {
    return SummaryName("TagsEntitySummaryForJSONFormatTest");
  }

  std::vector<std::string> Tags;
};

// ============================================================================
// NullEntitySummaryForJSONFormatTest - For null data checks
// ============================================================================

struct NullEntitySummaryForJSONFormatTest final : EntitySummary {
  SummaryName getSummaryName() const override {
    return SummaryName("NullEntitySummaryForJSONFormatTest");
  }
};

// ============================================================================
// UnregisteredEntitySummaryForJSONFormatTest - For missing FormatInfo checks
// ============================================================================

struct UnregisteredEntitySummaryForJSONFormatTest final : EntitySummary {
  SummaryName getSummaryName() const override {
    return SummaryName("UnregisteredEntitySummaryForJSONFormatTest");
  }
};

// ============================================================================
// MismatchedEntitySummaryForJSONFormatTest - For mismatched SummaryName checks
// ============================================================================

struct MismatchedEntitySummaryForJSONFormatTest final : EntitySummary {
  SummaryName getSummaryName() const override {
    return SummaryName("MismatchedEntitySummaryForJSONFormatTest_WrongName");
  }
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_UNITTESTS_SCALABLESTATICANALYSISFRAMEWORK_SERIALIZATION_JSONFORMATTEST_JSONFORMATTEST_H
